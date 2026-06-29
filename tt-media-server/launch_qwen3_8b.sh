#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# Qwen3-8B full model on a single P150 chip (chip 0 / port 8019 by default).
# Activate the tt-xla venv first, then run from this dir: `./launch_qwen3_8b.sh`.
# All defaults below are baked in (DEVICE_IDS=0, PREFILL_CHUNK_SIZE=2048, ENABLE_TRACE,
# gmu 0.30, full model). Override any inline, e.g. `PORT=8014 DEVICE_IDS=3 ./launch_qwen3_8b.sh`
# (coexist with other forge LLM servers) or `NUM_HIDDEN_LAYERS=1 ./launch_qwen3_8b.sh`
# (single-layer debug build).

export MODEL=${MODEL:-Qwen3-8B}
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
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-40960}  # Qwen3 native max_position_embeddings (no 64K)
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.30}
export TT_KV_POOL_GB=${TT_KV_POOL_GB:-32}
export OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL:-1}
export CPU_SAMPLING=${CPU_SAMPLING:-false}
export ENABLE_TRACE=${ENABLE_TRACE:-true}
# BFP8 KV cache (halves KV footprint vs bf16) so 64K context fits. "" => bf16.
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfp_bf8}
# On-device chunked-SDPA prefill chunk size; required to fit 64K context
# (without it the full-context prefill SDPA buffer OOMs the DRAM banks).
export PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE-2048}
# bf16 matmul dest accumulation (smaller buffers) to fit gmu 0.35; matches
# the tt-xla chunked-prefill sweep. "true" => fp32 dest accumulation.
export FP32_DEST_ACC_EN=${FP32_DEST_ACC_EN:-false}
# b1-prefill (tt-xla #5281): also compile a [1, n] prefill graph alongside the
# [32, n] b32 graph and route lone / small bursts of prefills to it.
#   MIN_NUM_SEQS=1             enables b1 (compile both graphs; pick per step).
#   PREFILL_BATCH_THRESHOLD=16 serves <=16 pending prefills on b1, serially, instead
#                             of one wasted-row b32 batch; >16 still batches to b32.
# Set MIN_NUM_SEQS= (empty) to fall back to the b32-only baseline. The b1 graph
# compiles lazily on the first lone-request prefill (precompile not yet ported).
export MIN_NUM_SEQS=${MIN_NUM_SEQS-1}
export PREFILL_BATCH_THRESHOLD=${PREFILL_BATCH_THRESHOLD-16}
# Pin this server to a single physical chip so all four can run concurrently.
# Use DEVICE_IDS, NOT TT_VISIBLE_DEVICES: the runner (setup_runner_environment)
# overwrites TT_VISIBLE_DEVICES with the worker's device_id derived from
# DEVICE_IDS, so setting TT_VISIBLE_DEVICES here has no effect.
export DEVICE_IDS=${DEVICE_IDS:-0}
# INFO logging so the resolved vLLM/plugin config is captured: look for the
# "[TT] Chunked prefill: capping max_num_batched_tokens N -> 2048" line and the
# engine-config line (enable_chunked_prefill, max_num_batched_tokens). Override
# LOG_LEVEL=warning to quiet it for normal runs.
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
LOG_LEVEL=${LOG_LEVEL:-info}

PORT=${PORT:-8019}
echo "Starting server: MODEL=$MODEL DEVICE=$DEVICE DEVICE_IDS=${DEVICE_IDS:-auto} PORT=$PORT"
cd "$(dirname "$0")"
uvicorn main:app --lifespan on --host 0.0.0.0 --port $PORT --log-level "$LOG_LEVEL"
