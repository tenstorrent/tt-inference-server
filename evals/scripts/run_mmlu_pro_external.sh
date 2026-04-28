#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Run MMLU-Pro against a remote server exposing OpenAI-compatible endpoints.
#
# By default this runs the full 12,032-question test split. Set MMLU_PRO_LIMIT or
# pass a positional LIMIT to run only a subset.
#
# Examples:
#   ./evals/scripts/run_mmlu_pro_external.sh
#   ./evals/scripts/run_mmlu_pro_external.sh 32
#   MMLU_PRO_LIMIT=256 ./evals/scripts/run_mmlu_pro_external.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
RUNNER="${HERE}/helper_external_lm_eval.sh"

LIMIT="${1:-${MMLU_PRO_LIMIT:-}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/mmlu_pro_${TIMESTAMP}}"

cmd=(
    "${RUNNER}"
    --task mmlu_pro
    --chat-api
    --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
fi

exec "${cmd[@]}"
