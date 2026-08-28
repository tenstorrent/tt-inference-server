#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Self-contained reproduction of the mini-swe-agent format-error loop.
#
# Bootstraps uv and a throwaway venv, then replays the prompts saved under a
# run's mini_sweagent/format_errors/ directory against the endpoint. Nothing
# from .workflow_venvs is needed, so this runs on a bare machine that only has
# the saved dumps and network access to the server.
#
# The failure being reproduced: the model intermittently collapses into a
# repetition loop, runs to its full max_tokens without ever emitting a bash
# tool call, and the harness discards the whole response. It is stochastic, so
# a single attempt often succeeds -- use REPEAT to sample.
#
# Usage:
#   scripts/repro_format_errors.sh <run_dir|format_errors_dir|dump.json> [extra args...]
#
# Environment overrides:
#   VENV_DIR       venv location (default /tmp/format_error_replay/venv)
#   PYTHON_VERSION interpreter for the venv (default 3.11)
#   REPEAT         attempts per dump (default 6)
#   CONCURRENCY    requests in flight (default 3)
#   OUTPUT_DIR     where replies are written (default <base>/format_error_replays)
#   OPENAI_API_KEY overrides the key saved in the run's model config
#   API_BASE       overrides the endpoint saved in the run's model config
#   MODEL          overrides the served model name
#
# Examples:
#   scripts/repro_format_errors.sh /path/to/swe_bench_verified_20260828T090918
#   REPEAT=10 scripts/repro_format_errors.sh /path/to/dump.json
#   scripts/repro_format_errors.sh /path/to/run --max-tokens 4096   # cost control test

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPLAY_SCRIPT="${REPLAY_SCRIPT:-${PROJECT_ROOT}/scripts/replay_format_errors.py}"

BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-/tmp/format_error_replay}"
VENV_DIR="${VENV_DIR:-${BOOTSTRAP_DIR}/venv}"
UV_INSTALL_DIR="${UV_INSTALL_DIR:-${BOOTSTRAP_DIR}/bin}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

REPEAT="${REPEAT:-6}"
CONCURRENCY="${CONCURRENCY:-3}"

if [[ $# -lt 1 ]]; then
    sed -n '6,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2
fi

BASE_DIR="$1"
shift

if [[ ! -e "${BASE_DIR}" ]]; then
    echo "ERROR: no such path: ${BASE_DIR}" >&2
    exit 1
fi
if [[ ! -f "${REPLAY_SCRIPT}" ]]; then
    echo "ERROR: replay script not found: ${REPLAY_SCRIPT}" >&2
    exit 1
fi

# ---------------------------------------------------------------- uv bootstrap

find_uv() {
    if [[ -n "${UV_BIN:-}" ]] && [[ -x "${UV_BIN}" ]]; then
        echo "${UV_BIN}"
    elif command -v uv >/dev/null 2>&1; then
        command -v uv
    elif [[ -x "${UV_INSTALL_DIR}/uv" ]]; then
        echo "${UV_INSTALL_DIR}/uv"
    fi
}

uv_bin="$(find_uv)"
if [[ -z "${uv_bin}" ]]; then
    echo "==> uv not found, installing to ${UV_INSTALL_DIR}"
    mkdir -p "${UV_INSTALL_DIR}"
    # The installer honours UV_INSTALL_DIR and needs no root.
    if ! curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="${UV_INSTALL_DIR}" INSTALLER_NO_MODIFY_PATH=1 sh; then
        echo "ERROR: could not install uv. Install it manually and re-run, or" >&2
        echo "       set UV_BIN=/path/to/uv." >&2
        exit 1
    fi
    uv_bin="$(find_uv)"
fi
if [[ -z "${uv_bin}" ]]; then
    echo "ERROR: uv still not on PATH after install" >&2
    exit 1
fi
echo "==> uv: ${uv_bin} ($("${uv_bin}" --version))"

# -------------------------------------------------------------- venv bootstrap

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "==> creating venv at ${VENV_DIR} (python ${PYTHON_VERSION})"
    "${uv_bin}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi

# The replay talks raw HTTP, so it needs no litellm, no minisweagent, no torch.
echo "==> installing dependencies"
VIRTUAL_ENV="${VENV_DIR}" "${uv_bin}" pip install --quiet \
    'requests>=2.31,<3' \
    'pyyaml>=6,<7'

venv_python="${VENV_DIR}/bin/python"

# ------------------------------------------------------------------- overrides

replay_args=()
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    replay_args+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${API_BASE:-}" ]]; then
    replay_args+=(--api-base "${API_BASE}")
fi
if [[ -n "${MODEL:-}" ]]; then
    replay_args+=(--model "${MODEL}")
fi
# OPENAI_API_KEY is read by the replay script straight from the environment.

echo "==> replaying ${BASE_DIR}"
echo "    repeat=${REPEAT} concurrency=${CONCURRENCY}"
echo

set -x
"${venv_python}" "${REPLAY_SCRIPT}" \
    "${BASE_DIR}" \
    --repeat "${REPEAT}" \
    --concurrency "${CONCURRENCY}" \
    "${replay_args[@]}" \
    "$@"
