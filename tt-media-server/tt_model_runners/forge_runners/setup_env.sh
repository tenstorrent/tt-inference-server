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
if ! command -v python3.12 >/dev/null 2>&1; then
    echo "Error: No suitable Python version found!"
    exit 1
fi

# WORKER_ENV=crank builds the tt-crank training env instead of the tt-forge one;
# the two pin different torch versions and cannot share a venv.
if [ "${WORKER_ENV:-forge}" = "crank" ]; then
    # tt-blacksmith's `env/activate --crank` builds the venv: the tt-crank wheel,
    # blacksmith's requirement files and blacksmith itself (editable). The venv lives
    # inside the checkout, so the checkout stays and venv-worker links to it. The
    # wheel download calls gh, which must be logged in (GH_TOKEN works).
    blacksmith_ref=06fe1f6d2ad68143eff178cb11a439f92385737d
    blacksmith_home="$(pwd)/tt-blacksmith"
    git init -q "${blacksmith_home}"
    git -C "${blacksmith_home}" fetch -q --depth 1 https://github.com/tenstorrent/tt-blacksmith.git "${blacksmith_ref}"
    git -C "${blacksmith_home}" checkout -q FETCH_HEAD
    (cd "${blacksmith_home}" && export PIP_NO_CACHE_DIR=1 && source env/activate --crank)
    rm -rf "${blacksmith_home}/env/wheels"
    rm -rf "${virtual_env_name}"
    ln -s "${blacksmith_home}/env/crank_env" "${virtual_env_name}"
    source ${virtual_env_name}/bin/activate

    # A second resolve on top of blacksmith's; pip check fails the build if it broke
    # a pin of tt-crank or blacksmith.
    pip install --no-cache-dir -r requirements.txt -r tt_model_runners/forge_training_runners/requirements.txt
    pip check
    # The wheel does not bundle the SFPI toolchain tt-metal's kernel JIT needs.
    tt-crank-install-sfpi
else
    echo "Using python3.12 for virtual environment..."
    python3.12 -m venv ${virtual_env_name}
    source ${virtual_env_name}/bin/activate

    # Set environment variables for vllm build
    export VLLM_TARGET_DEVICE="empty"

    pip install --upgrade pip

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
