"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def _sqls_by_iteration(history: list[dict]) -> list[str]:
    """Extract the SQL the agent held after each generate/revise step, in order.

    history entries from generate_sql / revise nodes carry a "sql" key. The
    verify entries don't, so we just skip them. Index 0 is the first attempt
    (generate_sql), index k>0 is the SQL after the k-th revise.
    """
    return [h["sql"] for h in history if "sql" in h]


def eval_one(question: dict, agent_url: str, tags: dict | None = None) -> dict:
    """Score one question via execution accuracy, per iteration.

    Calls the agent, then for every SQL the agent produced (first attempt +
    each revision) runs it against the target DB and compares its canonicalized
    rows to the gold query's rows. Records a correctness bool per iteration plus
    the final served result.
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    record: dict = {
        "db_id": db_id,
        "question": question["question"],
        "gold_sql": gold_sql,
        "gold_ok": gold_ok,
        "gold_error": gold_err,
    }

    try:
        resp = httpx.post(
            agent_url,
            json={"question": question["question"], "db": db_id, "tags": tags or {}},
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        record.update(
            agent_ok=False,
            agent_error=f"{type(e).__name__}: {e}",
            iterations=0,
            per_iteration_correct=[],
            final_sql="",
            final_correct=False,
        )
        return record

    history = data.get("history", [])
    iter_sqls = _sqls_by_iteration(history) or [data.get("sql", "")]

    per_iter: list[bool] = []
    for sql in iter_sqls:
        pred_ok, pred_rows, _ = run_sql(db_id, sql)
        per_iter.append(bool(gold_ok and pred_ok and matches(gold_rows, pred_rows)))

    record.update(
        agent_ok=bool(data.get("ok", False)),
        agent_error=data.get("error"),
        iterations=data.get("iterations", len(iter_sqls)),
        final_sql=data.get("sql", iter_sqls[-1] if iter_sqls else ""),
        per_iteration_correct=per_iter,
        final_correct=per_iter[-1] if per_iter else False,
    )
    return record


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {"n": 0}

    max_iters = max((len(r["per_iteration_correct"]) for r in results), default=0)

    pass_at_iteration: list[float] = []
    for k in range(max_iters):
        correct = 0
        for r in results:
            pic = r["per_iteration_correct"]
            if not pic:
                continue  # agent never produced a SQL -> wrong at every iteration
            idx = k if k < len(pic) else len(pic) - 1  # carry forward last attempt
            if pic[idx]:
                correct += 1
        pass_at_iteration.append(round(correct / n, 4))

    overall_correct = sum(1 for r in results if r["final_correct"])
    agent_failures = sum(1 for r in results if not r.get("agent_ok"))
    iterations = [r["iterations"] for r in results if r.get("iterations")]

    return {
        "n": n,
        "overall_pass_rate": round(overall_correct / n, 4),
        "overall_correct": overall_correct,
        "pass_rate_at_iteration": pass_at_iteration,
        "avg_iterations": round(sum(iterations) / len(iterations), 2) if iterations else 0,
        "max_iterations_observed": max_iters,
        "agent_failures": agent_failures,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    parser.add_argument("--model", default=None, help="Tag this run with a model name (e.g. Qwen3-30B-A3B)")
    parser.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
                        help="Repeatable; e.g. --tag experiment=perf_tuned --tag iteration=2")
    args = parser.parse_args()

    run_tags: dict = {}
    if args.model:
        run_tags["model"] = args.model
    for kv in args.tag:
        k, _, v = kv.partition("=")
        if k:
            run_tags[k] = v

    if args.out is None:
        if args.model:
            slug = args.model.replace("/", "_").replace(":", "_")
            out_path = ROOT / "results" / f"eval_{slug}.json"
        else:
            out_path = DEFAULT_OUT_FILE
    else:
        out_path = args.out

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")
    if run_tags:
        print(f"Tags: {run_tags}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url, tags=run_tags))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "tags": run_tags,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
