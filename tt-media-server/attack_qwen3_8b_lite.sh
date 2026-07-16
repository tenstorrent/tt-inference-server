#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# #4521 repro attack: fires the conc=64 continuous vllm bench serve load that has
# hung the Qwen3-8B server reliably (~40-85s) all through the 2026-07-15 overnight
# investigation. Assumes a server is ALREADY launched and warm on the given port
# (e.g. via launch_qwen3_8b_debug.sh / launch_qwen3_8b_ci.sh).
#
# Usage:
#   ./attack_qwen3_8b.sh [logfile] [port]
# Examples:
#   ./attack_qwen3_8b.sh                          # auto-named log, port 8019
#   ./attack_qwen3_8b.sh attack1.log               # custom log, port 8019
#   ./attack_qwen3_8b.sh attack1.log 8019          # explicit port
#
# Fixed ISL=1024/OSL=128 (random dataset with range-ratio 0 -> exact lengths, only
# per-request token content varies; content is reproducible run-to-run since vllm
# bench serve defaults to --seed 0). Deliberately NOT using identical prompt content
# across requests -- prefix caching is on server-side and would collapse most
# requests to cache hits, changing the load pattern.

LOGFILE=${1:-attack_lite_qwen3_8b_$(date +%Y%m%d_%H%M%S).log}
PORT=${2:-8019}

echo "Attacking 127.0.0.1:$PORT, logging to $LOGFILE"
OPENAI_API_KEY=your-secret-key timeout 30 /home/kmabee/tt-xla/venv/bin/vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions --model Qwen/Qwen3-8B \
  --host 127.0.0.1 --port "$PORT" \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --max-concurrency 64 --num-prompts 64 \
  --extra-body '{"truncate_prompt_tokens":"1024","max_tokens":128}' \
  |& tee "$LOGFILE"
