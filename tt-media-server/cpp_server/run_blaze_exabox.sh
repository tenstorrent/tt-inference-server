#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# Run the C++ media server with the real tt-llm-engine decode scheduler
# (LLM_DEVICE_BACKEND=pipeline_manager) on an exabox Blackhole Galaxy node.
#
# In pipeline_manager mode the server is a *client*: a separately-running model
# owns the devices and publishes host<->device socket descriptors to /dev/shm.
# The server connects to those sockets, so two things must line up:
#   1. The model must already be running and have published its descriptors
#      (e.g. demos/cli.py --external-io --io-socket-descriptor-prefix deepseek
#             --num-mesh-rows 4 --num-mesh-cols 2).
#   2. The server worker must OPEN the same physical devices the model uses,
#      otherwise H2DSocket::connect aborts with "Device id N not found in
#      cluster". The opened set is derived from DEVICE_IDS (NOT from
#      TT_VISIBLE_DEVICES, which worker_manager overrides). See DEVICE_IDS below.
#
# Build first with ./build_blaze_exabox.sh
#
# Usage:
#   ./run_blaze_exabox.sh                 # defaults below
#   PORT=8001 ./run_blaze_exabox.sh
#   DEVICE_IDS="(0,1,4,5,24,25,28,29)" DEVICE_MESH_SHAPE="4,2" ./run_blaze_exabox.sh
#   ./run_blaze_exabox.sh -t 16           # extra args are passed to the binary
#
# Override any of: PORT, DEVICE_IDS, DEVICE_MESH_SHAPE, BLAZE_SOCKET_DESCRIPTOR_PREFIX,
# OPENAI_API_KEY, LLM_DEVICE_BACKEND.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="${SCRIPT_DIR}/tt-llm-engine"
export TT_METAL_HOME="${TT_METAL_HOME:-${ENGINE_DIR}/tt-metal}"

# ── Tunables (override via env) ──────────────────────────────────────────
PORT="${PORT:-8000}"
# Backend selecting the real engine pipeline. Override to mock_pipeline to run
# without a model/devices for a quick smoke test.
export LLM_DEVICE_BACKEND="${LLM_DEVICE_BACKEND:-pipeline_manager}"
# DEVICE_IDS controls which physical chips the worker opens. ONE bracket pair =
# one worker; the ids inside are that worker's visible devices. The default
# matches a model launched on a 4x2 mesh on this Galaxy node (receiver -> dev 5).
export DEVICE_IDS="${DEVICE_IDS:-(0,1,4,5,24,25,28,29)}"
export DEVICE_MESH_SHAPE="${DEVICE_MESH_SHAPE:-4,2}"
# Socket descriptor prefix — must match the model's --io-socket-descriptor-prefix.
export BLAZE_SOCKET_DESCRIPTOR_PREFIX="${BLAZE_SOCKET_DESCRIPTOR_PREFIX:-deepseek}"

# ── Runtime library wiring ───────────────────────────────────────────────
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Node-local prefix (Drogon etc.) — Drogon links statically so this is usually
# unnecessary, but harmless and future-proofs against dynamic deps.
LOCAL_PREFIX="${LOCAL_PREFIX:-/data/${USER}/.local}"
[ -d "${LOCAL_PREFIX}/lib" ] && export LD_LIBRARY_PATH="${LOCAL_PREFIX}/lib:${LD_LIBRARY_PATH}"
# The engine .so must be preloaded so its symbols win over any in the binary.
ENGINE_SO="${ENGINE_DIR}/build-full/libtt_llm_engine.so.0"
export LD_PRELOAD="${ENGINE_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"

BINARY="${SCRIPT_DIR}/build/tt_media_server_cpp"

# ── Pre-flight checks ────────────────────────────────────────────────────
if [ ! -x "${BINARY}" ]; then
    echo "ERROR: server binary not found at ${BINARY}"
    echo "  Build it first: ./build_blaze_exabox.sh"
    exit 1
fi
if [ ! -f "${ENGINE_SO}" ]; then
    echo "ERROR: tt-llm-engine library not found at ${ENGINE_SO}"
    echo "  Build it first: ./build_blaze_exabox.sh"
    exit 1
fi

# In pipeline_manager mode, warn loudly if the model's socket descriptors are
# not present — connecting will otherwise hang/abort.
if [ "${LLM_DEVICE_BACKEND}" = "pipeline_manager" ]; then
    P="${BLAZE_SOCKET_DESCRIPTOR_PREFIX}"
    H2D_DESC="/dev/shm/tt_h2d_${P}_h2d.bin"
    D2H_DESC="/dev/shm/tt_d2h_${P}_d2h.bin"
    if [ ! -f "${H2D_DESC}" ] || [ ! -f "${D2H_DESC}" ]; then
        echo "WARNING: model socket descriptors not found for prefix '${P}':"
        echo "  expected ${H2D_DESC} and ${D2H_DESC}"
        echo "  Start the model first (it must publish these), e.g.:"
        echo "    python demos/cli.py --external-io --io-socket-descriptor-prefix ${P} \\"
        echo "      --num-mesh-rows 4 --num-mesh-cols 2"
        echo "  Continuing anyway; the worker will wait for the descriptors to appear."
    fi
fi

cat <<EOF
==========================================================================
  Starting TT Media Server (Blaze / ${LLM_DEVICE_BACKEND})
==========================================================================
  Port               : ${PORT}
  DEVICE_IDS         : ${DEVICE_IDS}
  DEVICE_MESH_SHAPE  : ${DEVICE_MESH_SHAPE}
  Socket prefix      : ${BLAZE_SOCKET_DESCRIPTOR_PREFIX}
  TT_METAL_HOME      : ${TT_METAL_HOME}
  LD_PRELOAD         : ${ENGINE_SO}
  API auth           : Authorization: Bearer ${OPENAI_API_KEY:-your-secret-key}$([ -z "${OPENAI_API_KEY:-}" ] && echo "  (default; set OPENAI_API_KEY to change)")
==========================================================================
EOF

exec "${BINARY}" -p "${PORT}" "$@"
