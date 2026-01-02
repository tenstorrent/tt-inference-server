#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# =============================================================================
# Inference Backend Selection
# =============================================================================
# INFERENCE_BACKEND environment variable controls which Python environment and
# library configuration is used:
#
#   metal (default): Uses tt-metal python_env with handcrafted model implementations
#                    Requires LD_LIBRARY_PATH pointing to tt-metal build libraries
#
#   xla:             Uses xla_venv with tt-xla compiled models (pjrt-plugin-tt)
#                    The pjrt-plugin-tt package bundles its own compatible tt-metal
#                    runtime, so LD_LIBRARY_PATH must not point to the local tt-metal
#                    build to avoid library version conflicts
# =============================================================================

if [ "$INFERENCE_BACKEND" = "xla" ]; then
    echo "Activating XLA inference backend: ${XLA_PYTHON_ENV_DIR}"
    source "${XLA_PYTHON_ENV_DIR}/bin/activate"

    # XLA backend uses bundled tt-metal runtime from pjrt-plugin-tt
    # Clear LD_LIBRARY_PATH to avoid conflicts with the local tt-metal build
    export LD_LIBRARY_PATH=""
else
    echo "Activating Metal inference backend: ${PYTHON_ENV_DIR}"
    source "${PYTHON_ENV_DIR}/bin/activate"

    # Metal backend requires tt-metal build libraries
    export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib"
fi

cd "${TT_METAL_HOME}/server/"
source ./run_uvicorn.sh
