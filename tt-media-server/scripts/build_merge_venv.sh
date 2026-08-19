#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Build the dedicated CPU venv used to merge LoRA adapters (see
# utils/adapter_merge_utils.py). Both versions are derived from their sources of
# truth so this venv never drifts from the versions that matter:
#   - transformers <- tt-vllm-plugin/pyproject.toml (the version vLLM serves
#                     with), parsed with tomllib so formatting can't break it.
#   - peft         <- the version *actually installed* in the forge venv (the
#                     env that WROTE the adapter), read back via
#                     importlib.metadata, so the merge reads its config with the
#                     exact same peft — no manual pin to keep in sync.
#
# Usage: build_merge_venv.sh <venv_dir> <tt-vllm-plugin pyproject.toml>
set -euo pipefail

venv_dir="$1"
plugin_pyproject="$2"

# This script lives in <tt-media-server>/scripts, so its parent is the app root.
server_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
merge_requirements="${server_dir}/adapter_merge_requirements.txt"

forge_python="${PYTHON_ENV_DIR:-${server_dir}/venv-worker}/bin/python"
if [ ! -x "${forge_python}" ]; then
    echo "ERROR: forge venv python not found at '${forge_python}'" >&2
    exit 1
fi

# transformers: the `transformers` dependency spec from the plugin's pyproject,
# parsed with tomllib so TOML formatting can't break it.
transformers_spec="$(python3.12 - "${plugin_pyproject}" <<'PY'
import sys
import tomllib

deps = tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]
print(next(d for d in deps if d.startswith("transformers")))
PY
)" || true

# peft: the version currently installed in the forge venv.
peft_version="$("${forge_python}" -c 'import importlib.metadata as m; print(m.version("peft"))' || true)"
peft_spec="peft==${peft_version}"

if [ -z "${transformers_spec}" ] || [ -z "${peft_version}" ]; then
    echo "ERROR: could not derive versions (transformers='${transformers_spec}', peft='${peft_spec}')" >&2
    exit 1
fi
echo "Merge venv versions -> transformers: ${transformers_spec} | peft: ${peft_spec}"

# Install the merge deps together with the two derived versions in a single resolve.
python3.12 -m venv "${venv_dir}"
"${venv_dir}/bin/pip" install --no-cache-dir --upgrade pip
"${venv_dir}/bin/pip" install --no-cache-dir \
    -r "${merge_requirements}" \
    "${transformers_spec}" "${peft_spec}"
