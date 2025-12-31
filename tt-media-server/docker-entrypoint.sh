#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Select Python environment based on USE_XLA_ENV
# Default: metal python_env, set USE_XLA_ENV=true for xla_venv
if [ "$USE_XLA_ENV" = "true" ]; then
    echo "Activating XLA Python environment: ${XLA_PYTHON_ENV_DIR}"
    source "${XLA_PYTHON_ENV_DIR}/bin/activate"

    # Clear LD_LIBRARY_PATH to avoid conflicts with tt-metal build
    # tt-xla bundles its own compatible tt-metal runtime libraries
    echo "Clearing LD_LIBRARY_PATH to use tt-xla bundled libraries"
    unset LD_LIBRARY_PATH
else
    echo "Activating Metal Python environment: ${PYTHON_ENV_DIR}"
    source "${PYTHON_ENV_DIR}/bin/activate"
fi

cd "${TT_METAL_HOME}/server/"
source ./run_uvicorn.sh

