# REPORT — LLM inference + observability

> **Status:** code + config complete. Sections marked `⟨FILL FROM LIVE RUN⟩` need
> numbers from the real `Qwen3-30B-A3B` endpoint on the H100 — they can only come
> from running the stack, per the README. Everything else is final.

---

## 1. Serving configuration (Phase 1)

Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, 30B total / ~3B active) on 1× H100 80GB.
Workload: ~1.5–3K-token prompts (schema + question), short SQL outputs, 2–3 dependent
calls per request, SLO **P95 end-to-end < 5s @ 10+ RPS**. Flags (in `scripts/start_vllm.sh`):

Setup issue: model default context length exceeds available KV cache

Qwen3-30B-A3B defaults to a max sequence length of 262,144 tokens, which requires 24 GiB of KV cache. On an H100 80GB, after loading model weights (~29 GiB), only ~12 GiB remains for KV cache — not enough. vLLM exits with:

ValueError: To serve at least one request with the model's max seq len (262144),
24.0 GiB KV cache is needed, which is larger than the available KV cache memory (12.35 GiB).

Fix: cap the context length to match the actual workload. Prompts in this assignment are ~1.5–3K tokens (schema + question) and SQL outputs are ~50–200 tokens — 4096 covers everything:

--max-model-len 4096

This reduces KV cache from 24 GiB to ~0.5 GiB, freeing the rest for concurrent requests.

I tested with subset_test_phase1.py and found that requests (subset) were completed within 5seconds (3s) 

P95: 3.89s ✓ under 5s SLO

Manual sanity check: `screenshots/vllm_manual_query.png`.

It is also clear that the model has no context that it needs to generate SQL. When served an eval question directly, it responded with a natural language answer rather than a SQL query. This is expected at this stage — the model has no system prompt to define its role and no schema to write against. Providing both is the work of Phase 3.

Example added under `screenshots/vllm_testing_with_eval.png`.

---

## 2. Observability dashboard (Phase 2)

`infra/grafana/provisioning/dashboards/serving.json`, organized into three rows that
answer "is it slow, and where?":

- **Latency:** e2e request latency p50/p95/p99 (5s threshold line); TTFT vs TPOT p95
  (prefill/queue-bound vs decode-bound); queue-wait p95; running vs waiting requests.
- **Throughput:** prompt vs generation tokens/s; finished requests/s by reason; running
  requests; generation tokens/s.
- **KV cache:** `gpu_cache_usage_perc` (with 80%/95% thresholds); preemptions/s +
  prefix-cache hit rate.

Reading guide: rising **TTFT + queue wait** while *running* is flat ⇒ admission/concurrency
bound (raise `--max-num-seqs` or cut prompt cost). Rising **TPOT** + **KV usage near 100%**
+ **preemptions > 0** ⇒ decode/KV bound (FP8, lower `max-model-len`, or fewer seqs).

Screenshot under load: `screenshots/grafana_serving.png` 

---

## 3. Agent design (Phase 3)

1.  Using Structured Output: Structured output is more reliable than string parsing — instead of regex-scraping SQL from markdown fences or JSON from prose, we bind each LLM call to a Pydantic model (SQLOutput, Verdict) so vLLM enforces the schema at the output layer. This eliminates a whole class of silent failures where the parser succeeds but extracts the wrong thing.

LangGraph: `attach_schema → generate_sql → execute → verify →` (`route_after_verify`) →
`revise → execute → verify` … capped at `MAX_ITERATIONS = 3`.

- **`generate_sql`** (provided shape): schema + question → one `\`\`\`sql\`\`\`` block.
- **`verify`** (vLLM call #2): always runs so the Langfuse waterfall shows a verify span.
  Fed the compact `ExecutionResult.render()`, returns parsed `{"ok", "issue"}`. Fires on
  the obvious cases: SQL errored, 0 rows when the question implies rows, columns that
  don't answer the question. Unparseable verdict → defaults `ok=true` (cap is the backstop).
- **`revise`** (vLLM call #3): gets failing SQL + result + the verifier's complaint.
- **`route_after_verify`**: end if `verify_ok` or `iteration >= MAX_ITERATIONS`, else revise.

Prompts (`agent/prompts.py`) keep the rules in a **stable system prefix** (no per-request
data) so vLLM prefix caching reuses it, with variable input in the user message.

---

## 4. Agent tracing (Phase 4)

Langfuse callback is wired in `agent/server.py` (initialized when `LANGFUSE_*` keys are
set; failures are not swallowed). Per-request `tags` are passed through as trace metadata
for Phase-6 filtering. Inspected trace (generate/verify/revise waterfall):
`screenshots/langfuse_trace.png`; tag list: `screenshots/langfuse_tags.png` 
I added experiment and model tags to track through prompt improvmements and testing different models. 
It was also helpful to observe token usage and latency of reasoning models such as openai_gpt-oss-120b.
---

## 5. Baseline eval (Phase 5)

`evals/run_eval.py` computes **execution accuracy**: runs the agent's SQL at each
iteration and the gold SQL against the target DB, compares canonicalized row sets
(sorted, stringified, `None`→`""`). Per-iteration pass rate uses carry-forward — if the
agent stopped at iteration *j < k*, its iteration-*k* result = its iteration-*j* result.

> **Note:** this baseline was run locally against `gpt-4o-mini` (OpenAI API) not the
> production `Qwen3-30B-A3B` endpoint. The purpose was to validate the eval pipeline and
> agent loop end-to-end before the H100 was available. These numbers are not representative
> of production quality — real pass rates must come from the 30B endpoint (see `results/eval_after_tuning.json`).

Run: `uv run python evals/run_eval.py --out results/eval_baseline.json`
(30 questions × ~2 calls ≈ 60 requests).

| Metric | Value |
|---|---|
| Overall pass rate | 36.67% (11/30) |
| Pass @ iter 0 / 1 / 2 | 26.67% → 33.33% → 36.67% |
| Avg iterations | 1.6 |
| Agent failures | 0 |
| Wall clock | 175s |

The verify→revise loop is earning its keep: pass rate goes up at every iteration (+10pp
from iter 0 to final). Starting at 26.67% after the first SQL attempt, each revision
recovered real failures — iter 1 added 6.7pp, iter 2 added another 3.3pp. If the loop
were doing nothing, all three numbers would be flat. Average iterations of 1.6 means most
questions needed more than one attempt but the full 3-iteration budget wasn't burned on
everything, which is a healthy balance.

The absolute 36.67% is expected for `gpt-4o-mini` on BIRD — it's a hard benchmark with
complex multi-join queries and `gpt-4o-mini` is not a strong text-to-SQL model. The
architecture validation is what matters here: zero agent failures, the pipeline runs
end-to-end, and the loop demonstrably helps.


**Iteration 2 — 120B open-source model via Nebius AI Studio:**

| Metric | Value |
|---|---|
| Overall pass rate | 30.0% (9/30) |
| Pass @ iter 0 / 1 / 2 | 30.0% → 30.0% → 30.0% |
| Avg iterations | 1.2 |
| Agent failures | 0 |

The loop added no value: pass rate was flat across all three iterations. The root cause is
self-confirmation bias — when the same large model acts as both generator and verifier, it
tends to approve its own outputs. The 120B model produces SQL that looks structurally
plausible (correct tables, non-zero rows, right column names) even when semantically wrong,
so the verifier rarely triggers a revision. Avg iterations of 1.2 confirms this: most
requests terminated after the first attempt. The architecture is sound but the verifier
needs to be independent of the generator to catch these failures.

**Prompt tuning experiment — tightening the zero-rows heuristic:**

A hypothesis was that the verifier's zero-rows rule was too aggressive — flagging "which"
and "list" questions that might legitimately return empty results. The rule was narrowed to
only trigger on questions naming specific entities or using superlatives ("the most X",
"the highest Y").

| Metric | Original | Tuned |
|---|---|---|
| Overall pass rate | 36.67% | 26.67% |
| Pass @ iter 0 / 1 / 2 | 26.67% → 33.33% → **36.67%** | 23.33% → 26.67% → **26.67%** |
| Avg iterations | 1.6 | 1.53 |
| Agent failures | 0 | 1 |

The change introduced a regression. The original rule was empirically correct for BIRD:
"which" and "list" questions on this benchmark almost always expect results — zero rows
means a bad JOIN or wrong filter, not a legitimately empty answer. Suppressing that trigger
stopped the loop from catching real failures, reducing its contribution from +10pp to
+3.3pp. The 1 agent failure also removed one previously-correct question, explaining the
drop in pass@0. The fix was to restore the original rule and add only a narrow exception
for questions that explicitly ask about absence ("students with no enrollment", "products
never sold").

From the Grafana dashboard during the Phase 5 eval run, I observed that KV cache utilization stays near zero between requests. This is expected: the eval script sends one request at a time sequentially, so there are no concurrent requests and no batching. Each request is processed independently — the GPU cache fills briefly during generation and drains immediately after. Between requests, the GPU sits idle. There is no prefix cache reuse across questions even though the system prompt and schema are repeated for every call.
---

## 6. Hitting the SLO (Phase 6)

1. based on previous observations and the low latency, I added prefix caching and allowing for batching sequences. 
first run : 

{
  "requested_rps": 10.0,
  "duration_seconds": 300,
  "wall_clock_seconds": 360.0042169589997,
  "total_requests": 3000,
  "achieved_rps": 8.3332357196851,
  "ok": 2578,
  "timeouts": 17,
  "http_errors": 376,
  "client_errors": 29,
  "latency_p50": 30.002315063000424,
  "latency_p95": 88.37885656900016,
  "latency_p99": 94.33577264200085,
  "latency_max": 109.2530676489996
}

- Request throughput panel — "length" finish reason is visible alongside "stop." Requests finishing at "length" means they're being cut off at max_tokens before the SQL is complete. Truncated SQL → execution error → unnecessary revise call → extra latency. Fix: increase max_tokens in the agent LLM call.
- KV cache panel — peaked at 36% with no preemptions. The GPU has significant headroom that is going unused. Fix: increase --max-num-seqs to admit more concurrent requests and fill that headroom.
- Running requests panel — bursty mountain shape, dropping low between waves. This confirms requests are not being batched smoothly — the queue empties and refills rather than staying full. Increasing --max-num-seqs directly addresses this by keeping more requests in flight at once.
- Token throughput panel — prompt tokens/s dominates, gen tokens/s is negligible. This is consistent with short or truncated outputs. Fixing max_tokens should raise the gen tokens line.

1. Increase output sequence length (max_tokens) → fixes the "length" truncations → reduces avoidable revise calls
2. Increase batching (--max-num-seqs) → fills the unused KV headroom → smooths the mountain pattern

I increased max_tokens to 512 and --max-num-seqs to 100. I then saw KV cache hit 100%, preemptions appearing, e2e degrading to 8–10s  I therefore reduced the max_num_seqs to 50. 

Target: **P95 e2e agent latency < 5s, 10+ RPS over 5 min.**

**Baseline:** ⟨FILL FROM LIVE RUN⟩.

> Reference point (another student's run, *not* mine, for calibration): achieved 9.28 RPS,
> p50 0.89s, **p95 2.66s (under SLO)**, p99 6.37s, max 52s, but **381 HTTP errors / 3000 (~13%)**.
> That shape — good P95 but a fat error tail and achieved RPS short of target — points at
> the server/queue rejecting or timing out under burst, not raw decode speed. First thing
> I'd check on my own run: `num_requests_waiting` + queue-wait p95 vs `gpu_cache_usage_perc`.

**Iteration log** (fill one line per change):

1. saw ⟨X⟩ → hypothesized ⟨Y⟩ → changed ⟨Z⟩ → result ⟨W⟩  (`grafana_before.png` / `grafana_after.png`)
2. …

**Final config vs SLO:** ⟨FILL⟩. **Did quality survive?** Re-run eval to
`results/eval_after_tuning.json` and compare to baseline ⟨FILL⟩.

---

## 7. Agent value & what I'd do with more time (Phase 7)

**Did the loop earn its keep?** Compare pass@iter0 vs pass@iter2 from §5. If they're equal,
the verify/revise loop is pure latency cost; if iter2 > iter0, the revise step is recovering
real failures. ⟨FILL with the actual delta and a one-line verdict⟩.

**With more time (specific):**
- Make `verify` cheaper/deterministic for the trivial cases (SQL error, 0 rows) and only
  spend an LLM call on the ambiguous ones — removes ~1 call/request from the hot path.
- Add few-shot schema-linking examples to `generate_sql` to lift iter-0 pass rate (cuts
  revises, which helps both quality *and* P95).
- Constrain decoding to SQL (grammar / structured output) to kill malformed-SQL retries.
- Cache rendered schemas across requests (already `lru_cache`d) and pre-warm the vLLM
  prefix cache per DB before load tests so cold-start prefill doesn't skew early P95.
