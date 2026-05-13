#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Stable entrypoint for DeepSeek-R1 endpoint eval suites. The Python runner
# handles suite orchestration and reporting; lm-eval execution still goes
# through helper_external_lm_eval.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/evals/deepseek_external_runner.py" "$@"
