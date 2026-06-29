#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# Llama-3.1-8B-Instruct on a single chip. Defaults to chip 0 / port 8012. Set
# DEVICE_IDS + PORT to run several forge LLM servers concurrently on a multi-chip
# host (see launch_falcon3_7b.sh, launch_llama_3b.sh, launch_qwen3_8b.sh).

export MODEL=${MODEL:-Llama-3.1-8B-Instruct}
export DEVICE=${DEVICE:-p150}
export API_KEY=${API_KEY:-your-secret-key}
export ENVIRONMENT=development
export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)/tt-metal}
# TT_MESH_GRAPH_DESC_PATH is optional: single-chip p150 does not need it. On a
# multi-chip host, export the board's mesh graph descriptor before launching:
#   export TT_MESH_GRAPH_DESC_PATH=<tt-metal>/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export IS_GALAXY=${IS_GALAXY:-False}
# Forge single-chip P150 LLM spec (mirrors workflows/model_specs/dev/cnn.yaml,
# commit 838eb7711: b32 / 64K / opt=1 / on-device sampling). Manual launches do
# not read the workflow YAML, so these must be set here.
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-65536}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.30}
export TT_KV_POOL_GB=${TT_KV_POOL_GB:-32}
export OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL:-1}
export CPU_SAMPLING=${CPU_SAMPLING:-false}
export ENABLE_TRACE=${ENABLE_TRACE:-true}
# BFP8 KV cache (halves KV footprint vs bf16) so 64K context fits. "" => bf16.
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfp_bf8}
# On-device chunked-SDPA prefill chunk size. With max_num_batched_tokens
# right-sized to batch*chunk (= 65536, see config/vllm_settings.py), 64K context
# FITS at gmu 0.30 with b1-prefill on — validated 2026-06-28 (head_dim 128, unlike
# Falcon3 which needs chunk 1024). Keep 2048.
export PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-2048}
# bf16 matmul dest accumulation (smaller buffers). "true" => fp32 dest accumulation.
export FP32_DEST_ACC_EN=${FP32_DEST_ACC_EN:-false}
# b1-prefill (tt-xla #5281): compile [1,n] alongside [32,n]. Matches the model_spec.
# Fits at 64K (does NOT cause the OOM). Set MIN_NUM_SEQS= (empty) for b32-only.
export MIN_NUM_SEQS=${MIN_NUM_SEQS-1}
export PREFILL_BATCH_THRESHOLD=${PREFILL_BATCH_THRESHOLD-16}
# Pin the chip with DEVICE_IDS (default chip 0). NOT TT_VISIBLE_DEVICES: the
# runner overwrites that with the worker's device_id, so it has no effect here.
# On a multi-chip host (e.g. QB2) set DEVICE_IDS to another chip to coexist.
export DEVICE_IDS=${DEVICE_IDS:-0}

echo "Starting server: MODEL=$MODEL DEVICE=$DEVICE DEVICE_IDS=${DEVICE_IDS:-auto} PORT=${PORT:-8012}"
cd "$(dirname "$0")"
PORT=${PORT:-8012}
uvicorn main:app --lifespan on --host 0.0.0.0 --port $PORT --log-level warning
