#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

set -eo pipefail

if [ "$1" != "--skip-venv" ]; then
    source "${TT_METAL_HOME}/python_env/bin/activate"
fi

# Short-term workaround: forge wheel's vllm_tt/worker.py hardcodes the device
# memory cap at 12 GB (vllm_tt/worker.py:236) — well below P150's 32 GB HBM —
# which artificially limits the KV-cache pool and blocks 16K seq len on 7B/8B
# models. When TT_KV_POOL_GB is set, swap the hardcode for an env-driven value
# at startup. No-op if the env var is unset or the expected pattern isn't
# found (e.g. after the upstream tt-xla fix lands and replaces the hardcode
# with the real device DRAM value, at which point this block self-deactivates
# and can be removed in a one-line revert).
if [ -n "${TT_KV_POOL_GB:-}" ]; then
    WORKER_PY=$(python3 -c "from importlib.util import find_spec; s=find_spec('vllm_tt'); print((s.submodule_search_locations[0] if s else '') + '/worker.py')" 2>/dev/null || true)
    if [ -f "$WORKER_PY" ] && grep -q "total_memory_size = 12 \* 1024\*\*3" "$WORKER_PY"; then
        sed -i 's|total_memory_size = 12 \* 1024\*\*3|total_memory_size = int(os.environ.get("TT_KV_POOL_GB", "12")) * 1024**3|' "$WORKER_PY"
        echo "[run_uvicorn] patched vllm_tt/worker.py: KV pool size now driven by TT_KV_POOL_GB=$TT_KV_POOL_GB"
    elif [ -f "$WORKER_PY" ]; then
        echo "[run_uvicorn] vllm_tt/worker.py at $WORKER_PY does not contain expected hardcode pattern; TT_KV_POOL_GB ignored"
    else
        echo "[run_uvicorn] could not locate vllm_tt/worker.py; TT_KV_POOL_GB ignored"
    fi
fi

# HF_HOME (a persistent cache_root path set by the launcher) isn't always
# writable by this uid. Fall back to a per-user cache so model download/load 
# never crashes at startup (sacrifices cache persistence for media model weights).
if [ -n "${HF_HOME:-}" ]; then
    if ( mkdir -p "${HF_HOME}" && touch "${HF_HOME}/.hf_write_test" ) 2>/dev/null; then
        rm -f "${HF_HOME}/.hf_write_test"
    else
        _hf_fallback="${HOME:-/home/container_app_user}/.cache/huggingface"
        echo "[run_uvicorn] HF_HOME=${HF_HOME} not writable; falling back to ${_hf_fallback} (weights will not persist)"
        mkdir -p "${_hf_fallback}"
        export HF_HOME="${_hf_fallback}"
    fi
fi

uvicorn --host 0.0.0.0 main:app --lifespan on --port "${SERVICE_PORT:-8000}"
