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

forge_requirements="tt_model_runners/forge_runners/requirements.txt"
blacksmith_ref=$(sed -n 's|^blacksmith @ git+https://github.com/tenstorrent/tt-blacksmith.git@\([0-9a-f]*\).*|\1|p' "${forge_requirements}")
if [ -z "${blacksmith_ref}" ]; then
    echo "Error: no tt-blacksmith git pin found in ${forge_requirements}."
    exit 1
fi

# tt-blacksmith's `env/activate --crank` builds the venv: the tt-crank wheel,
# blacksmith's requirement files and blacksmith itself (editable). The venv lives
# inside the checkout, so the checkout stays and venv-worker links to it. The
# wheel download calls gh, which must be logged in (GH_TOKEN works).
blacksmith_home="$(pwd)/tt-blacksmith"
rm -rf "${blacksmith_home}" "${virtual_env_name}"
git init -q "${blacksmith_home}"
git -C "${blacksmith_home}" fetch -q --depth 1 https://github.com/tenstorrent/tt-blacksmith.git "${blacksmith_ref}"
# Leave LFS files (blacksmith's test checkpoints) as pointers: the bare-URL fetch
# configures no remote for git-lfs to download them from, and nothing here uses them.
GIT_LFS_SKIP_SMUDGE=1 git -C "${blacksmith_home}" checkout -q FETCH_HEAD
(cd "${blacksmith_home}" && export PIP_NO_CACHE_DIR=1 && source env/activate --crank)
rm -rf "${blacksmith_home}/env/wheels"
ln -s "${blacksmith_home}/env/crank_env" "${virtual_env_name}"
source ${virtual_env_name}/bin/activate

# A second resolve on top of blacksmith's, minus blacksmith itself (already installed
# editable from the checkout); pip check fails the build if it broke a pin of
# tt-crank or blacksmith.
pip install --no-cache-dir -r requirements.txt -r <(grep -v '^blacksmith @' "${forge_requirements}")
pip check
# The wheel does not bundle the SFPI toolchain tt-metal's kernel JIT needs.
tt-crank-install-sfpi

echo "Setup complete in virtual environment ${virtual_env_name}."
echo "To activate, run: source ${virtual_env_name}/bin/activate"
