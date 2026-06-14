# Sahana — MLOps assignment to-do (due Tue June 17)

Goal: a complete, **honest, passing** submission. Pass first, optimize later.

---

## Status: code is done ✅
All the stubbed code is implemented and validated. What's left is running it on the
live VM and filling real numbers into `REPORT.md` (search for `⟨FILL FROM LIVE RUN⟩`).

| File | State |
|---|---|
| `agent/prompts.py` | ✅ generate / verify / revise prompts |
| `agent/graph.py` | ✅ verify_node, revise_node, route_after_verify |
| `evals/run_eval.py` | ✅ eval_one + summarize (per-iteration carry-forward) |
| `scripts/start_vllm.sh` | ✅ tuned flags + justifications |
| `infra/.../serving.json` | ✅ latency / throughput / KV-cache dashboard |
| `REPORT.md` | ✅ drafted, needs live numbers |

---

## Minimum to pass (in order)
- [ ] Model serving + 1 manual query screenshot (`screenshots/vllm_manual_query.png`)
- [ ] Agent answering over HTTP, with **one question that triggers a revise**
- [ ] Baseline eval JSON (`results/eval_baseline.json`)
- [ ] One load test at 10 RPS + **one** real tuning iteration (saw→hypothesized→changed→result)
- [ ] `REPORT.md` filled with real numbers + honest verdict

Anything past one tuning iteration is polish.

---

## Time-box
- [ ] **Sat 14** — Stack up: `uv sync`, `load_data.py`, `docker compose up -d`, start vLLM, 3 manual queries + screenshot. *(riskiest day — see blocker)*
- [ ] **Sun 15** — Agent end-to-end + Langfuse keys + baseline eval. Two screenshots.
- [ ] **Mon 16** — One load test @ 10 RPS, read dashboard, **one** tuning iteration, re-run eval.
- [ ] **Tue 17** — Fill `REPORT.md` with real numbers, write reflection, submit.

---

## Baseline run — exact commands (on the live VM)
```bash
uv sync
uv run python scripts/load_data.py          # downloads BIRD (~500MB)
docker compose up -d                          # Grafana/Prometheus/Langfuse
bash scripts/start_vllm.sh                    # own terminal; wait for "Application startup complete"
# new terminal:
uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001
# smoke test one question:
curl -s localhost:8001/answer -H 'Content-Type: application/json' \
  -d '{"question":"...", "db":"..."}' | python3 -m json.tool
# then:
uv run python evals/run_eval.py --out results/eval_baseline.json
uv run python load_test/driver.py --rps 10 --duration 300
```

**Blocker to watch:** if `uv sync` or the GPU fails on the VM (this dev box has no uv,
no running Docker, and GPU shows "Insufficient Permissions"), paste the error — that's
the Day-1 problem to solve first.

---

## Reading the dashboard under load
- Rising **TTFT + queue wait** while *running* is flat ⇒ concurrency-bound → raise `--max-num-seqs` or cut prompt cost.
- Rising **TPOT** + **KV usage near 100%** + **preemptions > 0** ⇒ decode/KV-bound → FP8, lower `--max-model-len`, or fewer seqs.
- First panels to check on the load test: **requests waiting + queue-wait p95** vs **KV cache usage %**.

---

## Lectures: essential vs skip
- **Essential:** KV cache & prefix caching, batching/concurrency (`max-num-seqs`), latency anatomy (TTFT vs decode/TPOT), reading p50/p95/p99.
- **Skip for now:** quantization internals, multi-GPU / tensor-parallel, Kubernetes, training.

---

## Reflection to write (Phase 7)
Baseline → what I tried → what improved → what didn't → what I learned. Cite the
per-iteration pass rate to say whether the verify/revise loop earned its keep. An
honest missed SLO with a metric-grounded diagnosis beats a green check you can't explain.
