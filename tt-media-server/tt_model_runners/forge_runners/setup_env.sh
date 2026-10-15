# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#!/bin/bash
set -eo pipefail
# filepath: /localdev/idjuric/tt-inference-server/tt-media-server/scripts/simple_setup.sh

echo "Reseting environment..."
unset TT_METAL_HOME
unset PYTHONPATH
unset WH_ARCH_YAML
unset ARCH_NAME

virtual_env_name="venv-worker"
# Create virtual environment with available Python version
if command -v python3.12 >/dev/null 2>&1; then
    echo "Using python3.12 for virtual environment..."
    python3.12 -m venv ${virtual_env_name}
else
    echo "Error: No suitable Python version found!"
    exit 1
fi

# Activate virtual environment
source ${virtual_env_name}/bin/activate

# Set environment variables for vllm build
export VLLM_TARGET_DEVICE="empty"

# Install requirements
pip install --upgrade pip

# WORKER_ENV=crank builds the tt-crank training env instead of the tt-forge one;
# the two pin different torch versions and cannot share a venv.
if [ "${WORKER_ENV:-forge}" = "crank" ]; then
    if [ -z "${TT_CRANK_WHEEL}" ]; then
        echo "Error: WORKER_ENV=crank needs TT_CRANK_WHEEL (URL or path of the tt-crank wheel)."
        exit 1
    fi
    # One resolve, so the wheel's pins are checked against every requirement file.
    pip install --no-cache-dir "${TT_CRANK_WHEEL}" \
        -r requirements.txt \
        -r tt_model_runners/forge_training_runners/requirements.txt
    # The wheel does not bundle the SFPI toolchain tt-metal's kernel JIT needs.
    tt-crank-install-sfpi
else
    # Install root requirements if exists
    if [ -f "requirements.txt" ]; then
        # Dissable pip's package cache, reducing disk usage during installation and Docker size
        pip install --no-cache-dir -r requirements.txt
    fi

    # Install forge requirements if exists
    if [ -f "tt_model_runners/forge_runners/requirements.txt" ]; then
        # Dissable pip's package cache, reducing disk usage during installation and Docker size
        pip install --no-cache-dir -r tt_model_runners/forge_runners/requirements.txt
        if [ -f "tt_model_runners/forge_runners/requirements-no-build-isolation.txt" ]; then
            pip install --no-cache-dir --no-deps --no-build-isolation -r tt_model_runners/forge_runners/requirements-no-build-isolation.txt
        fi
        tt-forge-install
    fi
fi

echo "Setup complete in virtual environment ${virtual_env_name}."
echo "To activate, run: source ${virtual_env_name}/bin/activate"
