#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Run AIME24 evals against a remote DeepSeek server exposing OpenAI-compatible
# endpoints. Used by run_aime24_all_external.sh and run_aime24_short_external.sh.
#
# See helper_external_lm_eval.sh for shared env vars and mechanics.

set -euo pipefail

TASK="${1:-}"
INCLUDE_PATH="${2:-}"

if [[ -z "${TASK}" ]]; then
    echo "usage: $0 <task_name> [include_path]" >&2
    exit 64
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${REPO_ROOT}/evals/scripts/helper_external_lm_eval.sh"

cmd=(
    "${RUNNER}"
    --task "${TASK}"
    --chat-api
    --num-fewshot 0
)

if [[ -n "${INCLUDE_PATH}" ]]; then
    cmd+=(--include-path "${INCLUDE_PATH}")
fi

exec "${cmd[@]}"
