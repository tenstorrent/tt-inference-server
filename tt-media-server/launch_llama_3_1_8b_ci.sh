#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Llama-3.1-8B-Instruct single-chip forge LLM launcher that mirrors the CI /
# production config exactly (workflows/model_specs/dev/cnn.yaml, FORGE spec):
# b32, 65536 context, chunk 2048, gmu 0.30, bfp8 weights+KV, opt=1, device
# sampling, trace, b1-prefill (min_num_seqs=1 / prefill_batch_threshold=16).
# Pinned to chip 0.
#
# Run from the tt-xla venv:
#   cd /home/kmabee/tt-xla && source venv/activate && \
#   cd /home/kmabee/tt-inference-server/tt-media-server && ./launch_llama_3_1_8b_ci.sh

export MODEL=${MODEL:-Llama-3.1-8B-Instruct}
export DEVICE=${DEVICE:-p150}
export API_KEY=${API_KEY:-your-secret-key}
export ENVIRONMENT=development
export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)/tt-metal}
export IS_GALAXY=${IS_GALAXY:-False}

# --- Llama-3.1-8B-Instruct FORGE spec, verbatim from cnn.yaml ---
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-65536}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.30}
export TT_KV_POOL_GB=${TT_KV_POOL_GB:-32}
export OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL:-1}
export CPU_SAMPLING=${CPU_SAMPLING:-false}
export ENABLE_TRACE=${ENABLE_TRACE:-true}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfp_bf8}
export PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-2048}
export MIN_NUM_SEQS=${MIN_NUM_SEQS:-1}
export PREFILL_BATCH_THRESHOLD=${PREFILL_BATCH_THRESHOLD:-16}

# Pin to chip 0 (DEVICE_IDS, not TT_VISIBLE_DEVICES — the runner derives
# TT_VISIBLE_DEVICES from the worker's device_id).
export DEVICE_IDS=${DEVICE_IDS:-'(0)'}

export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
LOG_LEVEL=${LOG_LEVEL:-info}
PORT=${PORT:-8019}

echo "Starting CI-config server: MODEL=$MODEL DEVICE_IDS=$DEVICE_IDS PORT=$PORT"
echo "  ctx=$MAX_MODEL_LENGTH b=$MAX_NUM_SEQS gmu=$GPU_MEMORY_UTILIZATION chunk=$PREFILL_CHUNK_SIZE kv=$KV_CACHE_DTYPE opt=$OPTIMIZATION_LEVEL cpu_sampling=$CPU_SAMPLING trace=$ENABLE_TRACE min_num_seqs=$MIN_NUM_SEQS threshold=$PREFILL_BATCH_THRESHOLD"
cd "$(dirname "$0")"
uvicorn main:app --lifespan on --host 0.0.0.0 --port "$PORT" --log-level "$LOG_LEVEL"
