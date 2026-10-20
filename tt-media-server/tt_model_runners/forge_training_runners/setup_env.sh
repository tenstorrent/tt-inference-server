#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# Builds venv-worker for the tt-crank training image (Dockerfile.crank).
# Run from the tt-media-server directory:
#   GH_TOKEN=... tt_model_runners/forge_training_runners/setup_env.sh <tt-blacksmith checkout dir>
#
# tt-blacksmith's `env/activate --crank`, at the ref pinned in requirements.txt next to
# this script, creates the venv: the tt-crank wheel (fetched from tt-mlir's nightly CI
# through gh, hence GH_TOKEN), blacksmith's pinned requirements and blacksmith itself
# (editable). The venv lives inside the checkout, so the checkout stays and venv-worker
# links to it.
set -euo pipefail

blacksmith_home="${1:?usage: $0 <tt-blacksmith checkout dir>}"
requirements="tt_model_runners/forge_training_runners/requirements.txt"
virtual_env_name="venv-worker"

blacksmith_ref="$(sed -n 's|^blacksmith @ git+https://github.com/tenstorrent/tt-blacksmith.git@\([0-9a-f]*\).*|\1|p' "${requirements}")"
if [ -z "${blacksmith_ref}" ]; then
    echo "Error: no tt-blacksmith git pin found in ${requirements}." >&2
    exit 1
fi

# LFS files (blacksmith's test checkpoints) stay pointers: the bare-URL fetch configures
# no remote for git-lfs to download them from.
git init -q "${blacksmith_home}"
git -C "${blacksmith_home}" fetch -q --depth 1 https://github.com/tenstorrent/tt-blacksmith.git "${blacksmith_ref}"
GIT_LFS_SKIP_SMUDGE=1 git -C "${blacksmith_home}" checkout -q FETCH_HEAD

# env/activate is written for interactive shells and reads unset variables.
(cd "${blacksmith_home}" && set +u && export PIP_NO_CACHE_DIR=1 && source env/activate --crank)
rm -rf "${blacksmith_home}/env/wheels"
ln -s "${blacksmith_home}/env/crank_env" "${virtual_env_name}"

# The rest goes on top; pip check fails the build if that broke a pin of tt-crank or blacksmith.
"${virtual_env_name}/bin/pip" install --no-cache-dir -r requirements.txt -r <(grep -v '^blacksmith @' "${requirements}")
"${virtual_env_name}/bin/pip" check

# The wheel does not bundle the SFPI toolchain tt-metal's kernel JIT needs.
"${virtual_env_name}/bin/tt-crank-install-sfpi"

echo "Setup complete in virtual environment ${virtual_env_name}."
