#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Bring-your-own uvicorn launcher for the Mistral-Small-3.1-24B-Instruct-2503
# Forge LLM (tensor-parallel) on galaxy (8x4 / 32-chip TP mesh).
#
# Defaults target BLACKHOLE galaxy (DEVICE=bh-galaxy), the serving bringup
# target. For Wormhole galaxy (where tt-xla Stage 1 was validated) run with
# DEVICE=galaxy OPTIMIZATION_LEVEL=1 CPU_SAMPLING=true.
#
# Prereqs (run on the galaxy host):
#   cd ~/tt-xla && source venv/activate         # tt-xla venv with the forge wheel
#   cd ~/tt-inference-server/tt-media-server
#   # Install ONLY the server deps. Do NOT `pip install -r requirements.txt` into
#   # the tt-xla venv: it hard-pins torch==2.7.1+cpu (via torchaudio) and clobbers
#   # tt-xla's torch 2.10.0+cpu, breaking torch_xla's _XLAC ABI (undefined symbol
#   # c10::TensorImpl::sym_is_non_overlapping_and_dense_custom).
#   pip install aiohttp colorama fastapi faster_fifo httpx huggingface_hub loguru \
#       prometheus-client prometheus-fastapi-instrumentator psutil pydantic-settings \
#       python-multipart tqdm uvicorn requests num2words pytest pytest-asyncio
#   SERVICE_PORT=8019 ./launch_mistral_small_31_24b.sh
#
# device_ids / device_mesh_shape / is_galaxy are NOT set here on purpose: settings.py
# derives them from ModelConfigs[(VLLMForge_MISTRAL_SMALL_31_24B, <device>)] once
# MODEL + DEVICE + MODEL_RUNNER are set (env would override, so leave them unset).
# Every value below is overridable from the environment.

set -eo pipefail

export MODEL="${MODEL:-Mistral-Small-3.1-24B-Instruct-2503}"
export DEVICE="${DEVICE:-bh-galaxy}"
export MODEL_RUNNER="${MODEL_RUNNER:-vllm_forge_mistral_small_31_24b}"

# Production config mirrors the tt-xla Stage 1 vLLM benchmark (tt-xla PR #5510):
# b32 / 8K context / GMU 0.65 / bfp8 weights+KV.
export MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
# Blackhole default: opt=0 (opt>=1 aborted in tt-mlir OpModel worker-grid
# validation on Blackhole, tt-mlir#8767/#8769). opt=0 allows on-device sampling,
# so CPU_SAMPLING=false. On Wormhole galaxy use OPTIMIZATION_LEVEL=1 CPU_SAMPLING=true.
export OPTIMIZATION_LEVEL="${OPTIMIZATION_LEVEL:-0}"
export ENABLE_TRACE="${ENABLE_TRACE:-true}"
export CPU_SAMPLING="${CPU_SAMPLING:-false}"

export SERVICE_PORT="${SERVICE_PORT:-8019}"

# TT_METAL_HOME is REQUIRED even for the BYO-uvicorn path: the plugin only sets
# TT_METAL_RUNTIME_ROOT, and tt-metal writes JIT kernels to $TT_METAL_HOME/built
# AND reads SOC descriptors from $TT_METAL_HOME/tt_metal/soc_descriptors. If unset,
# built resolves to /built (root) -> "Permission denied"; if pointed at an
# incomplete tt-metal copy -> "bad file: .../soc_descriptors/*_arch.yaml".
# Derive it from the plugin's OWN resolution (setup_tt_metal_home sets
# TT_METAL_RUNTIME_ROOT to the wheel copy if present else the third_party
# submodule) so it always matches the tt-metal the runtime actually uses.
# Exported here so the media-server worker subprocesses inherit it.
_tt_metal_root=$(python3 -c "import os, pjrt_plugin_tt as p; p.setup_tt_metal_home(); print(os.environ.get('TT_METAL_RUNTIME_ROOT',''))" 2>/dev/null || true)
if [ -z "${TT_METAL_HOME:-}" ]; then
    if [ -n "${_tt_metal_root}" ] && [ -d "${_tt_metal_root}/tt_metal/soc_descriptors" ]; then
        export TT_METAL_HOME="${_tt_metal_root}"
    else
        echo "[launch] WARNING: could not resolve a complete tt-metal (TT_METAL_RUNTIME_ROOT=${_tt_metal_root:-<empty>}); set TT_METAL_HOME manually to a tt-metal dir that contains tt_metal/soc_descriptors/"
    fi
fi

# Galaxy mesh descriptor, arch-matched to DEVICE and resolved from TT_METAL_HOME.
if [ -z "${TT_MESH_GRAPH_DESC_PATH:-}" ] && [ -n "${TT_METAL_HOME:-}" ]; then
    case "${DEVICE}" in
        bh-galaxy) _desc_name="single_bh_galaxy_mesh_graph_descriptor.textproto" ;;
        galaxy)    _desc_name="single_galaxy_mesh_graph_descriptor.textproto" ;;
        *)         _desc_name="" ;;
    esac
    if [ -n "${_desc_name}" ]; then
        _desc="${TT_METAL_HOME}/tt_metal/fabric/mesh_graph_descriptors/${_desc_name}"
        if [ -f "${_desc}" ]; then
            export TT_MESH_GRAPH_DESC_PATH="${_desc}"
        else
            echo "[launch] WARNING: descriptor not found at ${_desc}; leaving TT_MESH_GRAPH_DESC_PATH unset"
        fi
    fi
fi

echo "[launch] MODEL=${MODEL} DEVICE=${DEVICE} MODEL_RUNNER=${MODEL_RUNNER}"
echo "[launch] MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH} MAX_NUM_SEQS=${MAX_NUM_SEQS} GMU=${GPU_MEMORY_UTILIZATION}"
echo "[launch] OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL} ENABLE_TRACE=${ENABLE_TRACE} CPU_SAMPLING=${CPU_SAMPLING}"
echo "[launch] TT_METAL_HOME=${TT_METAL_HOME:-<unset>}"
echo "[launch] SERVICE_PORT=${SERVICE_PORT} TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-<unset>}"

# --skip-venv: assume the tt-xla venv is already active (see prereqs above).
exec ./run_uvicorn.sh --skip-venv
