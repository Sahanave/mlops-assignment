#!/usr/bin/env bash
#
# Start vLLM serving Qwen3-30B-A3B-Instruct for the text-to-SQL agent.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Workload this is tuned for (see README Phase 1):
#   - prompts ~1.5-3K tokens (schema + question)
#   - short, structured outputs (a single SQL statement, ~tens-to-low-hundreds of tokens)
#   - ~2-3 dependent LLM calls per user request (generate -> verify -> maybe revise)
#   - SLO target: P95 end-to-end agent latency < 5s at 10+ RPS
#
# Hardware: 1x H100 80GB. Model: Qwen3-30B-A3B (MoE, 30B total / ~3B active params).
#
# Each flag has a one-line justification. The two flags most worth revisiting in
# Phase 6 are --quantization fp8 and --max-num-seqs (see notes).

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    `#Phase-1 improvements`
    `# Prompts are <=~3K and outputs are short; capping context at 4096 shrinks per-request` \
    `# KV cache footprint, which directly buys more concurrent sequences and also frees up memory.` \
    --max-model-len 4096 \
    `# Leave ~10% headroom for activations/CUDA graphs while maximizing KV cache space.` \
    --gpu-memory-utilization 0.90 
    `# Cap concurrent sequences so we batch for throughput without queueing so deep that` \
    `# P95 blows past 5s. 64 is a starting point - this is the main Phase 6 latency lever.` \
    --max-num-seqs 64 \
    `# THE big win for this workload: the schema prefix is identical across the 2-3 calls of` \
    `# a request (and reused across questions on the same DB), so prefix caching skips` \
    `# recomputing 1.5-3K prompt tokens on the verify/revise calls.` \
    --enable-prefix-caching 
