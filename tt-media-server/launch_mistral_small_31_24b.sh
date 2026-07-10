#!/bin/bash

# cd ~/tt-xla && source venv/activate
# cd ~/tt-inference-server/tt-media-server
#  # Install ONLY the server deps. Do NOT `pip install -r requirements.txt` into tt-xla venv.
# pip install aiohttp colorama fastapi faster_fifo httpx huggingface_hub loguru \
#       prometheus-client prometheus-fastapi-instrumentator psutil pydantic-settings \
#       python-multipart tqdm uvicorn requests num2words pytest pytest-asyncio
# SERVICE_PORT=8019 ./launch_mistral_small_31_24b.sh

set -eo pipefail

export MODEL=Mistral-Small-3.1-24B-Instruct-2503
export DEVICE=bh-galaxy
export MODEL_RUNNER="${MODEL_RUNNER:-vllm_forge_mistral_small_31_24b}"

# Production config mirrors the tt-xla Stage 1 vLLM benchmark (tt-xla PR #5510):
# b32 / 8K context / GMU 0.65 / bfp8 weights+KV.
export MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
# Validated Stage 1 config (matches tt-xla _mistral_small_31_tp_config on this
# galaxy): opt=1 with on-CPU sampling. At opt=1 the tt-mlir optimizer picks a
# sharded DRAM layout for the prefill logits matmul; opt=0 falls back to a naive
# layout that needs a full [max_len x vocab] buffer per DRAM bank -> ~32GB/chip
# OOM in prefill postprocess. opt>=1 + trace also REQUIRES cpu_sampling=true
# (TTConfig rejects opt>=1 AND trace AND cpu_sampling=false), so both are set
# together. These match the runner's own Python defaults.
export OPTIMIZATION_LEVEL="${OPTIMIZATION_LEVEL:-1}"
export ENABLE_TRACE="${ENABLE_TRACE:-true}"
export CPU_SAMPLING="${CPU_SAMPLING:-true}"

export SERVICE_PORT="${SERVICE_PORT:-8019}"

# tt-metal path resolution. Two distinct consumers:
#   - JIT kernel build dir: $TT_METAL_HOME/built (unset -> /built -> Permission denied)
#   - SOC descriptors: read from the plugin's TT_METAL_RUNTIME_ROOT, which the
#     worker RE-RESOLVES (env may not propagate) to the plugin-local tt-metal if
#     that dir exists. If that copy is INCOMPLETE -> "bad file:
#     .../pjrt_plugin_tt/tt-metal/.../soc_descriptors/*_arch.yaml".
# So we must pin BOTH vars to the SAME complete tt-metal and reject an
# incomplete one. Prefer an explicit TT_METAL_RUNTIME_ROOT already in the env
# (e.g. from venv/activate); else use the plugin's resolution.
_tt_metal_root="${TT_METAL_RUNTIME_ROOT:-}"
if [ -z "${_tt_metal_root}" ]; then
    _tt_metal_root=$(python3 -c "import os, pjrt_plugin_tt as p; p.setup_tt_metal_home(); print(os.environ.get('TT_METAL_RUNTIME_ROOT',''))" 2>/dev/null || true)
fi
# Reject an incomplete tt-metal: require at least one *_arch.yaml SOC descriptor.
if [ -n "${_tt_metal_root}" ] && ls "${_tt_metal_root}/tt_metal/soc_descriptors/"*_arch.yaml >/dev/null 2>&1; then
    export TT_METAL_HOME="${TT_METAL_HOME:-${_tt_metal_root}}"
    export TT_METAL_RUNTIME_ROOT="${_tt_metal_root}"
else
    echo "[launch] WARNING: no complete tt-metal found (root=${_tt_metal_root:-<empty>} lacks tt_metal/soc_descriptors/*_arch.yaml)."
    echo "[launch]   Find one:  find ~/tt-xla -path '*tt_metal/soc_descriptors/blackhole_140_arch.yaml' -size +0c"
    echo "[launch]   Then set:  export TT_METAL_RUNTIME_ROOT=<that tt-metal root>  (and re-run)"
    echo "[launch]   If the WORKER still re-resolves to an incomplete plugin-local copy, symlink it:"
    echo "[launch]     cd ~/tt-xla/python_package/pjrt_plugin_tt && mv tt-metal tt-metal.bak && ln -s <complete tt-metal root> tt-metal"
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
echo "[launch] TT_METAL_HOME=${TT_METAL_HOME:-<unset>} TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-<unset>}"
echo "[launch] SERVICE_PORT=${SERVICE_PORT} TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-<unset>}"

# --skip-venv: assume the tt-xla venv is already active (see prereqs above).
exec ./run_uvicorn.sh --skip-venv
