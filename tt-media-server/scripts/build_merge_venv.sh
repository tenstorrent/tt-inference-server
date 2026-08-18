#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Build the dedicated CPU venv used to merge LoRA adapters (see
# utils/adapter_merge_utils.py). The pins come from their sources of truth so
# this venv never drifts from the versions that matter:
#   - transformers <- tt-vllm-plugin/pyproject.toml (the version vLLM serves with)
#   - peft         <- forge_runners/requirements.txt (the version that wrote the
#                     adapter, so the merge reads its config back with the same peft)
#
# Usage: build_merge_venv.sh <venv_dir> <tt-vllm-plugin pyproject.toml>
set -euo pipefail

venv_dir="$1"
plugin_pyproject="$2"
server_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

transformers_spec="$(sed -nE 's/.*"(transformers[^"]*)".*/\1/p' "${plugin_pyproject}" | head -1)"
peft_spec="$(sed -nE 's/^[[:space:]]*(peft[^#[:space:]]*).*/\1/p' \
    "${server_dir}/tt_model_runners/forge_runners/requirements.txt" | head -1)"

if [ -z "${transformers_spec}" ] || [ -z "${peft_spec}" ]; then
    echo "ERROR: could not extract pins (transformers='${transformers_spec}', peft='${peft_spec}')" >&2
    exit 1
fi
echo "Merge venv pins -> transformers: ${transformers_spec} | peft: ${peft_spec}"

python3.12 -m venv "${venv_dir}"
"${venv_dir}/bin/pip" install --no-cache-dir --upgrade pip
"${venv_dir}/bin/pip" install --no-cache-dir \
    -r "${server_dir}/adapter_merge_requirements.txt" \
    "${transformers_spec}" "${peft_spec}"
