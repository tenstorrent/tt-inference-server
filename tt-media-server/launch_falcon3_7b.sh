#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# Falcon3-7B-Instruct on a single chip. Pinned to chip 0 / port 8001 so it can
# run alongside the other single-chip forge LLM servers (see launch_llama_8b.sh,
# launch_llama_3b.sh, launch_qwen3_8b.sh).

export TT_MESH_GRAPH_DESC_PATH="/home/kmabee/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
export MODEL=${MODEL:-Falcon3-7B-Instruct}
export DEVICE=${DEVICE:-p150}
export API_KEY=${API_KEY:-your-secret-key}
export ENVIRONMENT=development
export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)/tt-metal}
export IS_GALAXY=${IS_GALAXY:-False}
# Forge single-chip P150 LLM spec (mirrors workflows/model_specs/dev/cnn.yaml,
# commit 838eb7711: b32 / 64K / opt=1 / on-device sampling). Manual launches do
# not read the workflow YAML, so these must be set here.
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-32768}  # Falcon3 native max_position_embeddings (no 64K)
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.35}
export TT_KV_POOL_GB=${TT_KV_POOL_GB:-32}
export OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL:-1}
export CPU_SAMPLING=${CPU_SAMPLING:-false}
export ENABLE_TRACE=${ENABLE_TRACE:-true}
# BFP8 KV cache (halves KV footprint vs bf16) so 64K context fits. "" => bf16.
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfp_bf8}
# On-device chunked-SDPA prefill chunk size; required to fit 64K context
# (without it the full-context prefill SDPA buffer OOMs the DRAM banks).
export PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-2048}
# Pin this server to a single physical chip so all four can run concurrently.
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0}
[ -n "$DEVICE_IDS" ] && export DEVICE_IDS

echo "Starting server: MODEL=$MODEL DEVICE=$DEVICE TT_VISIBLE_DEVICES=$TT_VISIBLE_DEVICES DEVICE_IDS=${DEVICE_IDS:-auto}"
cd "$(dirname "$0")"
PORT=${PORT:-8001}
uvicorn main:app --lifespan on --host 0.0.0.0 --port $PORT --log-level warning
