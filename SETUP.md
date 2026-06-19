# SETUP.md — bring the stack up on the VM

A tick-through checklist for getting everything running on the live VM. Do these in
order; each step has a "verify" so you catch a break before it cascades. For the
higher-level plan and what to submit, see `sahaan_to_do.md`.

---

## Local development (macOS, no H100)

Use this to build and test the agent on your laptop against OpenAI before touching the VM.
`vllm` has no macOS wheels, so install only the agent deps.

**1. Install agent dependencies into `.venv`:**
```bash
.venv/bin/python -m ensurepip
.venv/bin/python -m pip install \
  "langgraph>=1.0,<2.0" "langchain>=1.0,<2.0" "langchain-openai>=1.0,<2.0" \
  "langfuse>=4.0,<5.0" "fastapi>=0.115,<1.0" "uvicorn[standard]>=0.30,<1.0" \
  "pydantic>=2.0,<3.0" "python-dotenv>=1.0,<2.0" \
  "httpx>=0.27,<1.0" "tqdm>=4.66,<5.0" "datasets>=2.20"
```

**2. Configure `.env` to point at OpenAI:**
```bash
cp .env.example .env
```
Uncomment and fill in these three lines in `.env`:
```
VLLM_BASE_URL=https://api.openai.com/v1
VLLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

**3. Download BIRD data:**
```bash
.venv/bin/python scripts/load_data.py
```
Verify: `wc -l evals/eval_set.jsonl` → 30

**4. Start the agent server:**
```bash
PYTHONPATH=. .venv/bin/uvicorn agent.server:app --host 0.0.0.0 --port 8001 --reload
```
(`PYTHONPATH=.` makes `import agent` resolve; `--reload` restarts on file saves.)

**5. Smoke-test:**
```bash
curl -s localhost:8001/health
# → {"status":"ok"}

curl -s -X POST localhost:8001/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the coordinates location of the circuits for Australian grand prix?","db":"formula_1"}' \
  | python3 -m json.tool
```

---

## 0. Connect + forward ports
- [ ] SSH into the VM, forwarding all five ports:
  ```bash
  ssh -L 3000:localhost:3000 -L 9090:localhost:9090 -L 3001:localhost:3001 \
      -L 8000:localhost:8000 -L 8001:localhost:8001 <user>@<vm-host>
  ```
  *(Or use VSCode/Cursor Remote-SSH + the Ports panel — forward 3000, 9090, 3001, 8000, 8001.)*
- [ ] **Verify:** prompt shows the VM hostname.

## 1. Repo + dependencies
- [ ] `cd mlops-assignment && git pull` (get the latest committed code)
- [ ] `uv sync`
- [ ] **Verify:** `uv run python -c "import vllm, langgraph, fastapi; print('deps ok')"`
  - ⚠️ If `uv` isn't installed: `curl -LsSf https://astral.sh/uv/install.sh | sh` then re-open the shell.

## 2. Environment file
- [ ] `cp .env.example .env`
- [ ] Put your `HF_TOKEN` in `.env` (needed to download the model).
- [ ] Leave `LANGFUSE_*` blank for now — you fill those in step 6.
- [ ] **Verify:** `grep HF_TOKEN .env` shows your token.

## 3. Load BIRD data
- [ ] `uv run python scripts/load_data.py`   *(downloads ~500MB, takes a few min)*
- [ ] **Verify:**
  ```bash
  ls data/bird/*.sqlite | head        # several .sqlite files
  wc -l evals/eval_set.jsonl           # 30
  wc -l load_test/perf_pool.jsonl      # ~1500
  ```

## 4. Observability stack (Docker)
- [ ] `docker compose up -d`
- [ ] **Verify:** `docker compose ps` — prometheus, grafana, langfuse-web, postgres,
  clickhouse, redis, minio all `Up` (langfuse takes ~1 min to finish migrating).
- [ ] In your laptop browser:
  - [ ] Prometheus → http://localhost:9090
  - [ ] Grafana → http://localhost:3000 (admin / admin)
  - [ ] Langfuse → http://localhost:3001
  - ⚠️ If a URL won't load, the port-forward is almost always the culprit (step 0).

## 5. Serve the model (vLLM)
- [ ] In its **own terminal:** `bash scripts/start_vllm.sh`
- [ ] Wait for `Application startup complete` (first run downloads the model — slow).
- [ ] **Verify:** `curl -s localhost:8000/v1/models | python3 -m json.tool` lists the model.
  - ⚠️ OOM at load? Confirm `--quantization fp8` is active; if it still won't fit, drop
    `--gpu-memory-utilization` to 0.85 or `--max-model-len` to 3072.
  - ⚠️ GPU "Insufficient Permissions" / not visible? `nvidia-smi` should list the GPU; if
    not, that's a VM/driver access issue to fix before anything else works.
- [ ] **Screenshot:** vLLM startup + one manual query → `screenshots/vllm_manual_query.png`

## 6. Wire Langfuse keys
- [ ] In Langfuse UI (http://localhost:3001): sign up locally → create/open a project →
  Settings → API Keys → create one.
- [ ] Paste `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` into `.env`
  (keep `LANGFUSE_HOST=http://localhost:3001`).

## 7. Start the agent server
- [ ] In its **own terminal:** `uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001`
- [ ] **Verify:** `curl -s localhost:8001/health` → `{"status":"ok"}`
- [ ] Smoke-test one question (pick a real `db_id` from `data/bird/`):
  ```bash
  curl -s localhost:8001/answer -H 'Content-Type: application/json' \
    -d '{"question":"How many ...?", "db":"<db_id>"}' | python3 -m json.tool
  ```
  Look for `sql`, `rows`, `iterations`, and a `history` array.
- [ ] **Verify a revise fires:** try 5 questions from `evals/eval_set.jsonl` until one
  shows a `revise` entry in `history` (iterations ≥ 2).
- [ ] **Screenshots:** a Langfuse trace with the generate/verify/(revise) waterfall →
  `screenshots/langfuse_trace.png`; the trace list with your tags → `screenshots/langfuse_tags.png`

## 8. You're ready for the runs
Stack is up. Now move to `sahaan_to_do.md`:
- [ ] Baseline eval → `results/eval_baseline.json` (screenshot Grafana during it →
  `screenshots/grafana_eval_run.png`)
- [ ] Load test @ 10 RPS for 300s → read dashboard → one tuning iteration
- [ ] Fill the `⟨FILL FROM LIVE RUN⟩` blanks in `REPORT.md`

---

### Quick teardown / restart
```bash
docker compose down            # stop o11y stack (volumes persist)
docker compose down -v         # also wipe Grafana/Langfuse/Prometheus data
# vLLM and the agent: just Ctrl-C their terminals
```

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


## Reading the dashboard under load
- Rising **TTFT + queue wait** while *running* is flat ⇒ concurrency-bound → raise `--max-num-seqs` or cut prompt cost.
- Rising **TPOT** + **KV usage near 100%** + **preemptions > 0** ⇒ decode/KV-bound → FP8, lower `--max-model-len`, or fewer seqs.
- First panels to check on the load test: **requests waiting + queue-wait p95** vs **KV cache usage %**.




Example commands for the H100 run:
# Experiment 1 — baseline vLLM config
uv run python evals/run_eval.py --model Qwen3-30B-A3B --tag experiment=baseline

# Experiment 2 — perf-tuned (restart vLLM with start_vllm_final.sh first)
uv run python evals/run_eval.py --model Qwen3-30B-A3B --tag experiment=perf_tuned

# Experiment 3 — Nebius hosted model (set .env, restart agent server)
uv run python evals/run_eval.py --model Qwen3-235B-A22B --tag experiment=nebius