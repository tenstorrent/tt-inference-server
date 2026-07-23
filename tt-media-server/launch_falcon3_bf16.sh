#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Falcon3-7B-Instruct single-chip forge LLM launcher -- bf16 dtype config.
# Bare-metal uvicorn (no docker), same forge P150 spec as the baseline
# (b32, 32768 ctx, chunk 1024, opt=1, device sampling, trace, b1-prefill) but
# with bf16 weights + bf16 KV cache instead of bfp_bf8. bf16 ~2x the weight
# footprint, so gmu is lowered to 0.20 to fit. No fidelity override by default.
#
# To add the fidelity knobs (the "bf16 + hifi4 + fp32" config), pass them in:
#   MATH_FIDELITY=hifi4 FP32_DEST_ACC_EN=true ./launch_falcon3_bf16.sh
#
# Run from a tt-xla venv, and tee the output yourself, e.g.:
#   cd <tt-xla checkout> && source venv/activate
#   TT_INFERENCE_SERVER_ROOT=<tt-inference-server> TT_METAL_HOME=<tt-metal> \
#     ./tt-media-server/launch_falcon3_bf16.sh |& tee falcon_server_bf16.log
#
# TT_METAL_HOME must point to a tt-metal checkout containing the P150 mesh
# descriptor; TT_MESH_GRAPH_DESC_PATH is derived from it (override to be exact).
set -eo pipefail  # NOT -u: venv/activate references vars unset until sourced

TT_INFERENCE_SERVER_ROOT=${TT_INFERENCE_SERVER_ROOT:-$HOME/tt-inference-server}
export TTXLA_LOGGER_LEVEL=${TTXLA_LOGGER_LEVEL:-DEBUG}

export TT_METAL_HOME=${TT_METAL_HOME:-$(pwd)/tt-metal}
export TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto}
export MODEL=${MODEL:-Falcon3-7B-Instruct}
export DEVICE=${DEVICE:-p150}
export API_KEY=${API_KEY:-your-secret-key}
export ENVIRONMENT=development
export IS_GALAXY=${IS_GALAXY:-False}

# Falcon3-7B FORGE spec (cnn.yaml). Overridable so we can sweep knobs.
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-32768}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.20}
export TT_KV_POOL_GB=${TT_KV_POOL_GB:-32}
export OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL:-1}
export CPU_SAMPLING=${CPU_SAMPLING:-false}
export ENABLE_TRACE=${ENABLE_TRACE:-true}
export PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-1024}
export MIN_NUM_SEQS=${MIN_NUM_SEQS:-1}
export PREFILL_BATCH_THRESHOLD=${PREFILL_BATCH_THRESHOLD:-16}

# bf16 precision: WEIGHT_DTYPE="" => bf16 weights (no on-device quantization);
# KV_CACHE_DTYPE=none => bf16 KV cache. ${VAR:-} keeps WEIGHT_DTYPE empty.
export WEIGHT_DTYPE=${WEIGHT_DTYPE:-}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-none}

# Fidelity overrides: UNSET by default (plugin per-op defaults). Only export
# when NON-EMPTY -- an empty FP32_DEST_ACC_EN would be read as False (a
# graph-wide override, tt-xla #5116 hazard), not "unset". The `if` form (not
# `[ ] &&`) avoids tripping `set -e` when the var is absent.
if [ -n "${MATH_FIDELITY:-}" ]; then export MATH_FIDELITY; fi
if [ -n "${FP32_DEST_ACC_EN:-}" ]; then export FP32_DEST_ACC_EN; fi

export DEVICE_IDS=${DEVICE_IDS:-'(0)'}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
LOG_LEVEL=${LOG_LEVEL:-info}
PORT=${PORT:-8019}

echo "Starting Falcon3-7B-Instruct (bf16): PORT=$PORT DEVICE_IDS=$DEVICE_IDS ctx=$MAX_MODEL_LENGTH b=$MAX_NUM_SEQS gmu=$GPU_MEMORY_UTILIZATION chunk=$PREFILL_CHUNK_SIZE min_num_seqs=$MIN_NUM_SEQS threshold=$PREFILL_BATCH_THRESHOLD math_fidelity='${MATH_FIDELITY:-<unset>}' fp32_dest_acc_en='${FP32_DEST_ACC_EN:-<unset>}' weight_dtype='${WEIGHT_DTYPE:-bf16}' kv_cache_dtype='$KV_CACHE_DTYPE'"
echo "tt-inference-server root=$TT_INFERENCE_SERVER_ROOT  TT_METAL_HOME=$TT_METAL_HOME"

cd "$TT_INFERENCE_SERVER_ROOT/tt-media-server"
exec uvicorn main:app --lifespan on --host 0.0.0.0 --port "$PORT" --log-level "$LOG_LEVEL"
