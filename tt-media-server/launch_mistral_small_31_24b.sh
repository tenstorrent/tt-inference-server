#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Bring-your-own uvicorn launcher for the Mistral-Small-3.1-24B-Instruct-2503
# Forge LLM (tensor-parallel) on WH galaxy (galaxy-wh-6u, 8x4 / 32-chip mesh).
#
# Prereqs (run on a galaxy-wh-6u host):
#   cd ~/tt-xla && source venv/activate         # tt-xla venv with the forge wheel
#   cd ~/tt-inference-server/tt-media-server
#   pip install -r requirements.txt             # one-time: server deps into the venv
#   SERVICE_PORT=8019 ./launch_mistral_small_31_24b.sh
#
# device_ids / device_mesh_shape / is_galaxy are NOT set here on purpose: settings.py
# derives them from ModelConfigs[(VLLMForge_MISTRAL_SMALL_31_24B, GALAXY)] once
# MODEL + DEVICE + MODEL_RUNNER are set (env would override, so leave them unset).
# Every value below is overridable from the environment.

set -eo pipefail

export MODEL="${MODEL:-Mistral-Small-3.1-24B-Instruct-2503}"
export DEVICE="${DEVICE:-galaxy}"
export MODEL_RUNNER="${MODEL_RUNNER:-vllm_forge_mistral_small_31_24b}"

# Production config validated in tt-xla Stage 1 (tt-xla PR #5510).
export MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
# opt=1 validated on WH galaxy; opt>=1 + trace requires CPU sampling (the runner
# reads these). For on-device sampling, set OPTIMIZATION_LEVEL=0 and CPU_SAMPLING=false.
export OPTIMIZATION_LEVEL="${OPTIMIZATION_LEVEL:-1}"
export ENABLE_TRACE="${ENABLE_TRACE:-true}"
export CPU_SAMPLING="${CPU_SAMPLING:-true}"

export SERVICE_PORT="${SERVICE_PORT:-8019}"

# WH galaxy 8x4 mesh descriptor, resolved from the installed pjrt_plugin_tt so
# this works on any host/venv (not a hardcoded path). Harmless if the full-mesh
# path doesn't strictly require it.
if [ -z "${TT_MESH_GRAPH_DESC_PATH:-}" ]; then
    _plugin_base=$(python3 -c "from importlib.util import find_spec; import os; s=find_spec('pjrt_plugin_tt'); print(os.path.dirname(s.origin) if s else '')" 2>/dev/null || true)
    if [ -n "${_plugin_base}" ]; then
        _desc="${_plugin_base}/tt-metal/tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto"
        if [ -f "${_desc}" ]; then
            export TT_MESH_GRAPH_DESC_PATH="${_desc}"
        else
            echo "[launch] WARNING: descriptor not found at ${_desc}; leaving TT_MESH_GRAPH_DESC_PATH unset"
        fi
    else
        echo "[launch] WARNING: pjrt_plugin_tt not importable; leaving TT_MESH_GRAPH_DESC_PATH unset"
    fi
fi

echo "[launch] MODEL=${MODEL} DEVICE=${DEVICE} MODEL_RUNNER=${MODEL_RUNNER}"
echo "[launch] MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH} MAX_NUM_SEQS=${MAX_NUM_SEQS} GMU=${GPU_MEMORY_UTILIZATION}"
echo "[launch] OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL} ENABLE_TRACE=${ENABLE_TRACE} CPU_SAMPLING=${CPU_SAMPLING}"
echo "[launch] SERVICE_PORT=${SERVICE_PORT} TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-<unset>}"

# --skip-venv: assume the tt-xla venv is already active (see prereqs above).
exec ./run_uvicorn.sh --skip-venv
