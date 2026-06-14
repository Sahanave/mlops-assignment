# SETUP.md — bring the stack up on the VM

A tick-through checklist for getting everything running on the live VM. Do these in
order; each step has a "verify" so you catch a break before it cascades. For the
higher-level plan and what to submit, see `sahaan_to_do.md`.

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
