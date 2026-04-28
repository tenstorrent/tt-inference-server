#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Print an AIME24 generation-length report from an lm_eval output directory.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
DEFAULT_LM_EVAL_VENV="${REPO_ROOT}/.workflow_venvs/.venv_evals_common"
PYTHON_BIN="${PYTHON:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x "${LM_EVAL_VENV:-${DEFAULT_LM_EVAL_VENV}}/bin/python" ]]; then
        PYTHON_BIN="${LM_EVAL_VENV:-${DEFAULT_LM_EVAL_VENV}}/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "python3 not found" >&2
        exit 127
    fi
fi

exec "${PYTHON_BIN}" "${HERE}/helper_aime24_length_report.py" "$@"
