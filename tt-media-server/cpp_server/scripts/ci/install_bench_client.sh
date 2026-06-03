#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

set -euo pipefail

VENV_PATH=".venv"
PYTHON_BIN=""
PACKAGES=(vllm "transformers<5.6")

usage() {
    cat <<'EOF'
Usage: install_bench_client.sh [options]

Options:
  --venv PATH             Virtualenv path (default: .venv)
  --python PYTHON         Python interpreter for uv (optional)
  --package PACKAGE       Extra package to install. Can be repeated.
  --no-default-packages   Do not install vllm and transformers<5.6.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv) VENV_PATH="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --package) PACKAGES+=("$2"); shift 2 ;;
        --no-default-packages) PACKAGES=(); shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "$PYTHON_BIN" ]]; then
    uv venv --python "$PYTHON_BIN" "$VENV_PATH"
else
    uv venv "$VENV_PATH"
fi

if [[ "${#PACKAGES[@]}" -gt 0 ]]; then
    uv pip install --python "${VENV_PATH}/bin/python" "${PACKAGES[@]}"
fi

venv_abs="$(cd "$(dirname "$VENV_PATH")" && pwd)/$(basename "$VENV_PATH")"
echo "PATH=${venv_abs}/bin:${PATH}" >> "$GITHUB_ENV"
