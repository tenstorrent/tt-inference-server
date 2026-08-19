#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Build the dedicated CPU venv used to merge LoRA adapters (see
# utils/adapter_merge_utils.py). The two versions that must match the rest of
# the stack are resolved by scripts/derive_merge_versions.py:
#   - transformers <- tt-vllm-plugin/pyproject.toml (the version vLLM serves with)
#   - peft         <- the version installed in the forge venv (the env that wrote
#                     the adapter, so the merge reads its config back the same)
#
# Usage: build_merge_venv.sh <venv_dir> <tt-vllm-plugin pyproject.toml>
set -euo pipefail

venv_dir="$1"
plugin_pyproject="$2"

# This script lives in <tt-media-server>/scripts, so its parent is the app root.
server_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
merge_requirements="${server_dir}/adapter_merge_requirements.txt"
derive="${server_dir}/scripts/derive_merge_versions.py"

# The forge venv (transformers + the peft that trained the adapter). Prefer
# the path the Dockerfile exports; fall back to the conventional location.
forge_python="${PYTHON_ENV_DIR:-${server_dir}/venv-worker}/bin/python"

transformers_spec="$(python3.12 "${derive}" transformers "${plugin_pyproject}")"
peft_spec="$(python3.12 "${derive}" peft "${forge_python}")"
echo "Merge venv versions -> transformers: ${transformers_spec} | peft: ${peft_spec}"

# Install the merge deps together with the two derived versions in a single resolve.
python3.12 -m venv "${venv_dir}"
"${venv_dir}/bin/pip" install --no-cache-dir --upgrade pip
"${venv_dir}/bin/pip" install --no-cache-dir \
    -r "${merge_requirements}" \
    "${transformers_spec}" "${peft_spec}"
