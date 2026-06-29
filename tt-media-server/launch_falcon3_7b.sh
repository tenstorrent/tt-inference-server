#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# Falcon3-7B-Instruct on a single chip. Defaults to chip 0 / port 8011. Set
# DEVICE_IDS + PORT to run several forge LLM servers concurrently on a multi-chip
# host (see launch_llama_8b.sh, launch_llama_3b.sh, launch_qwen3_8b.sh).

export MODEL=${MODEL:-Falcon3-7B-Instruct}
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
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-32768}  # Falcon3 native max_position_embeddings (no 64K)
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.30}
export TT_KV_POOL_GB=${TT_KV_POOL_GB:-32}
export OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL:-1}
export CPU_SAMPLING=${CPU_SAMPLING:-false}
export ENABLE_TRACE=${ENABLE_TRACE:-true}
# BFP8 KV cache (halves KV footprint vs bf16) so 64K context fits. "" => bf16.
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfp_bf8}
# On-device chunked-SDPA prefill chunk size. MUST be 1024 here (not 2048 like
# Qwen3-8B): Falcon3 has head_dim=256 (vs 128), so the b32 prefill SDPA buffer
# at chunk 2048 is ~3.0 GB and OOMs DRAM at trace-capture (no contiguous block).
# 1024 halves it to fit. Not gmu/b1-prefill-related (both ruled out by identical
# OOM numbers); chunk size is the lever that scales this buffer.
export PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-1024}
# bf16 matmul dest accumulation (smaller buffers). "true" => fp32 dest accumulation.
export FP32_DEST_ACC_EN=${FP32_DEST_ACC_EN:-false}
# b1-prefill (tt-xla #5281): compile [1,n] alongside [32,n], serve <=16 pending
# prefills on b1 serially. Matches the Qwen3-8B / model_spec config. Set
# MIN_NUM_SEQS= (empty) to fall back to the b32-only baseline.
export MIN_NUM_SEQS=${MIN_NUM_SEQS-1}
export PREFILL_BATCH_THRESHOLD=${PREFILL_BATCH_THRESHOLD-16}
# Pin the chip with DEVICE_IDS (default chip 0). NOT TT_VISIBLE_DEVICES: the
# runner overwrites that with the worker's device_id, so it has no effect here.
# On a multi-chip host (e.g. QB2) set DEVICE_IDS to another chip to coexist.
export DEVICE_IDS=${DEVICE_IDS:-0}

echo "Starting server: MODEL=$MODEL DEVICE=$DEVICE DEVICE_IDS=${DEVICE_IDS:-auto} PORT=${PORT:-8011}"
cd "$(dirname "$0")"
PORT=${PORT:-8011}
uvicorn main:app --lifespan on --host 0.0.0.0 --port $PORT --log-level warning
