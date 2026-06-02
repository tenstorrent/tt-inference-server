#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# Unified Blaze build: builds everything needed to run the C++ media server
# with the real decode scheduler from tt-llm-engine.
#
# Mirrors Dockerfile.blaze but runs locally. Steps:
#   1. init the tt-llm-engine submodule (+ tt-metal)
#   2. build tt-metal and create its python_env
#   3. build tt-llm-engine with metal backend -> tt-llm-engine/build-full/libtt_llm_engine.so
#   4. build the C++ server with --blaze (links against tt-metal)
#
# Re-running is cheap: each step is skipped if its output already exists.
# Use --fresh-engine to force a rebuild of tt-metal + tt-llm-engine.
#
# Usage:
#   ./build_blaze.sh
#   ./build_blaze.sh --fresh-engine
#
# After building, run the server with:
#   export LD_LIBRARY_PATH="$(pwd)/tt-llm-engine/tt-metal/build/lib:$LD_LIBRARY_PATH"
#   LD_PRELOAD="$(pwd)/tt-llm-engine/build-full/libtt_llm_engine.so.0" \
#     ./build/tt_media_server_cpp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="${SCRIPT_DIR}/tt-llm-engine"
export TT_METAL_HOME="${ENGINE_DIR}/tt-metal"
export PYTHON_ENV_DIR="${TT_METAL_HOME}/python_env"

FRESH_ENGINE=0
[ "${1:-}" = "--fresh-engine" ] && FRESH_ENGINE=1

# tokenizers-cpp (built by both tt-llm-engine and the cpp_server) needs Rust.
if ! command -v cargo >/dev/null 2>&1 && [ -f "${HOME}/.cargo/env" ]; then
    . "${HOME}/.cargo/env"
fi

# ── 1. tt-llm-engine submodule (+ tt-metal) ──────────────────────────────
if [ ! -f "${TT_METAL_HOME}/CMakeLists.txt" ]; then
    echo "=== Initializing tt-llm-engine submodule ==="
    git -C "${SCRIPT_DIR}" submodule update --init --recursive tt-llm-engine
    git -C "${ENGINE_DIR}" submodule update --init --recursive
fi

# ── 2. tt-metal + python_env ─────────────────────────────────────────────
METAL_CONFIG="${TT_METAL_HOME}/build/lib/cmake/tt-metalium/tt-metaliumConfig.cmake"
if [ "${FRESH_ENGINE}" = 1 ] || [ ! -f "${METAL_CONFIG}" ]; then
    echo "=== Building tt-metal ==="
    (cd "${TT_METAL_HOME}" && ./build_metal.sh --disable-profiler)
fi
if [ ! -d "${PYTHON_ENV_DIR}" ]; then
    echo "=== Creating tt-metal python_env ==="
    (cd "${TT_METAL_HOME}" && ./create_venv.sh)
fi

# ── 3. tt-llm-engine (with metal backend) ────────────────────────────────
if [ "${FRESH_ENGINE}" = 1 ] || [ ! -f "${ENGINE_DIR}/build-full/libtt_llm_engine.so" ]; then
    echo "=== Building tt-llm-engine (--ds-full) ==="
    (cd "${ENGINE_DIR}" && ./setup.sh --ds-full)
fi

# ── 4. C++ media server (--blaze) ────────────────────────────────────────
echo "=== Building C++ media server (--blaze) ==="
# shellcheck source=/dev/null
source "${PYTHON_ENV_DIR}/bin/activate"
"${SCRIPT_DIR}/build.sh" --blaze

cat <<EOF

==========================================================================
  Blaze build complete.
==========================================================================
Run the server with the real decode scheduler:

  export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib:\${LD_LIBRARY_PATH:-}"
  LD_PRELOAD="${ENGINE_DIR}/build-full/libtt_llm_engine.so.0" \\
    ${SCRIPT_DIR}/build/tt_media_server_cpp

If Drogon / other deps live in a local prefix, prepend it too:
  export LD_LIBRARY_PATH="/data/\${USER}/.local/lib:\${LD_LIBRARY_PATH}"
EOF
