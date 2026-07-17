#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# gemma-4-31b-it TP (4-chip, (1,4) mesh) forge LLM launcher that mirrors the
# CI / production config (workflows/model_specs/dev/cnn.yaml, gemma-4-31b-it
# FORGE spec on P300X2): b32, 32768 context, chunk 1024, gmu 0.5, bfp8
# weights+KV, opt=1, device sampling, trace, b1-prefill (min_num_seqs=1 /
# prefill_batch_threshold=16). Pinned to all 4 chips as a single TP group.
#
# Run from the tt-xla-2 venv (this box's tt-xla-2 build has the Blackhole
# P300 opt>=1 fix already; see vllm_forge_gemma4_31b.py for details):
#   cd /home/kmabee/tt-xla-2 && source venv/activate && \
#   cd /home/kmabee/tt-inference-server/tt-media-server && ./launch_gemma4_31b_it_ci.sh

export TT_MESH_GRAPH_DESC_PATH="/home/kmabee/tt-xla-2/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
#export TTXLA_LOGGER_LEVEL=DEBUG

export MODEL=${MODEL:-gemma-4-31b-it}
export DEVICE=${DEVICE:-p300x2}
export API_KEY=${API_KEY:-your-secret-key}
export ENVIRONMENT=development
export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)/tt-metal}
export IS_GALAXY=${IS_GALAXY:-False}

# --- gemma-4-31b-it FORGE spec, verbatim from cnn.yaml ---
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-32768}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.5}
export OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL:-1}
export CPU_SAMPLING=${CPU_SAMPLING:-false}
export ENABLE_TRACE=${ENABLE_TRACE:-true}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfp_bf8}
export PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-1024}
export MIN_NUM_SEQS=${MIN_NUM_SEQS:-1}
export PREFILL_BATCH_THRESHOLD=${PREFILL_BATCH_THRESHOLD:-16}

unset FP32_DEST_ACC_EN

# Pin to all 4 chips as a single TP group (DEVICE_IDS, not TT_VISIBLE_DEVICES
# — the runner derives TT_VISIBLE_DEVICES from the worker's device_id).
export DEVICE_IDS=${DEVICE_IDS:-'(0,1,2,3)'}

export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
LOG_LEVEL=${LOG_LEVEL:-info}
PORT=${PORT:-8020}

echo "Starting CI-config server: MODEL=$MODEL DEVICE_IDS=$DEVICE_IDS PORT=$PORT"
echo "  ctx=$MAX_MODEL_LENGTH b=$MAX_NUM_SEQS gmu=$GPU_MEMORY_UTILIZATION chunk=$PREFILL_CHUNK_SIZE kv=$KV_CACHE_DTYPE opt=$OPTIMIZATION_LEVEL cpu_sampling=$CPU_SAMPLING trace=$ENABLE_TRACE min_num_seqs=$MIN_NUM_SEQS threshold=$PREFILL_BATCH_THRESHOLD"
cd "$(dirname "$0")"
uvicorn main:app --lifespan on --host 0.0.0.0 --port "$PORT" --log-level "$LOG_LEVEL"
