#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# Qwen3-4B full model on a single P150 chip (chip 0 / port 8010 by default).
# Activate the tt-xla venv first, then run from this dir: `./launch_qwen3_4b.sh`.
# Mirrors launch_qwen3_8b.sh (same forge single-chip config); only the model and
# default port differ. Override inline, e.g. `PORT=8010 DEVICE_IDS=0 ./launch_qwen3_4b.sh`.

export MODEL=${MODEL:-Qwen3-4B}
export DEVICE=${DEVICE:-p150}
export API_KEY=${API_KEY:-your-secret-key}
export ENVIRONMENT=development
export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)/tt-metal}
# TT_MESH_GRAPH_DESC_PATH is optional: single-chip p150 does not need it. On a
# multi-chip host, export the board's mesh graph descriptor before launching:
#   export TT_MESH_GRAPH_DESC_PATH=<tt-metal>/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export IS_GALAXY=${IS_GALAXY:-False}
# Forge single-chip P150 LLM spec (mirrors workflows/model_specs/dev/cnn.yaml
# Qwen3-4B entry). Manual launches do not read the workflow YAML, so set here.
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-40960}  # Qwen3 native max_position_embeddings
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.30}
export TT_KV_POOL_GB=${TT_KV_POOL_GB:-32}
export OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL:-1}
export CPU_SAMPLING=${CPU_SAMPLING:-false}
export ENABLE_TRACE=${ENABLE_TRACE:-true}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfp_bf8}  # BFP8 KV (weights BFP8 too)
# On-device chunked-SDPA prefill chunk size. 2048 fits (head_dim 128, like Qwen3-8B).
export PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE-2048}
export FP32_DEST_ACC_EN=${FP32_DEST_ACC_EN:-false}
# b1-prefill (tt-xla #5281): compile [1,n] alongside [32,n]. Matches the model_spec.
# Set MIN_NUM_SEQS= (empty) for the b32-only baseline.
export MIN_NUM_SEQS=${MIN_NUM_SEQS-1}
export PREFILL_BATCH_THRESHOLD=${PREFILL_BATCH_THRESHOLD-16}
# Pin via DEVICE_IDS, NOT TT_VISIBLE_DEVICES (the runner overwrites the latter).
export DEVICE_IDS=${DEVICE_IDS:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
LOG_LEVEL=${LOG_LEVEL:-info}

PORT=${PORT:-8010}
echo "Starting server: MODEL=$MODEL DEVICE=$DEVICE DEVICE_IDS=${DEVICE_IDS:-auto} PORT=$PORT"
cd "$(dirname "$0")"
uvicorn main:app --lifespan on --host 0.0.0.0 --port $PORT --log-level "$LOG_LEVEL"
