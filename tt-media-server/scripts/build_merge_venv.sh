#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Build the dedicated CPU venv used to merge LoRA adapters (see
# utils/adapter_merge_utils.py). Both pins are read from their sources of truth
# so this venv never drifts from the versions that matter:
#   - transformers <- tt-vllm-plugin/pyproject.toml (the version vLLM serves with)
#   - peft         <- forge_runners/requirements.txt (the version that wrote the
#                     adapter, so the merge reads its config back with the same peft)
#
# Usage: build_merge_venv.sh <venv_dir> <tt-vllm-plugin pyproject.toml>
set -euo pipefail

venv_dir="$1"
plugin_pyproject="$2"

# This script lives in <tt-media-server>/scripts, so its parent is the app root.
server_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
forge_requirements="${server_dir}/tt_model_runners/forge_runners/requirements.txt"
merge_requirements="${server_dir}/adapter_merge_requirements.txt"

# Grab the exact pin string out of each source file:
#   "transformers==4.55.0"   (quoted TOML dependency) -> transformers==4.55.0
#   peft==0.20.0 # comment   (requirements line)      -> peft==0.20.0
transformers_spec="$(grep -oE '"transformers[^"]*"' "${plugin_pyproject}" | tr -d '"' | head -1)"
peft_spec="$(grep -oE '^peft[^ #]*' "${forge_requirements}" | head -1)"

if [ -z "${transformers_spec}" ] || [ -z "${peft_spec}" ]; then
    echo "ERROR: could not extract pins (transformers='${transformers_spec}', peft='${peft_spec}')" >&2
    exit 1
fi
echo "Merge venv pins -> transformers: ${transformers_spec} | peft: ${peft_spec}"

# Install the merge deps together with the two derived pins in a single resolve.
python3.12 -m venv "${venv_dir}"
"${venv_dir}/bin/pip" install --no-cache-dir --upgrade pip
"${venv_dir}/bin/pip" install --no-cache-dir \
    -r "${merge_requirements}" \
    "${transformers_spec}" "${peft_spec}"
