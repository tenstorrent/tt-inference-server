#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Run DeepSeek-style AIME24 pass@1 estimation: 16 independent full-AIME runs,
# then aggregate all sample JSONL files into one pass@1 summary.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
RUNNER="${HERE}/helper_aime24_external.sh"
AGGREGATOR="${HERE}/helper_aggregate_pass1.py"

RUNS="${AIME_PASS1_RUNS:-16}"
TASK="${AIME_PASS1_TASK:-r1_aime24}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_OUTPUT="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/${TASK}_pass1x${RUNS}_${TIMESTAMP}}"
BASE_MODEL_SEED="${MODEL_SEED:-1234000}"

if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || [[ "${RUNS}" -lt 1 ]]; then
    echo "AIME_PASS1_RUNS must be a positive integer, got: ${RUNS}" >&2
    exit 64
fi

mkdir -p "${ROOT_OUTPUT}"

include_args=()
if [[ "${TASK}" == "r1_aime24_short" ]]; then
    include_args=("${REPO_ROOT}/evals/custom_tasks/r1_aime24_short")
fi

echo "Running ${TASK} pass@1 estimate"
echo "Samples per problem: ${RUNS}"
echo "Output dir: ${ROOT_OUTPUT}"

for run_idx in $(seq 1 "${RUNS}"); do
    run_label="$(printf 'sample_%02d' "${run_idx}")"
    run_output="${ROOT_OUTPUT}/${run_label}"
    run_seed=$((BASE_MODEL_SEED + run_idx - 1))

    echo
    echo "==> ${run_label}/${RUNS} (MODEL_SEED=${run_seed})"

    MODEL_SEED="${run_seed}" \
    OUTPUT_DIR="${run_output}" \
        "${RUNNER}" "${TASK}" "${include_args[@]}"
done

echo
"${AGGREGATOR}" "${ROOT_OUTPUT}" --task "${TASK}" --expected-runs "${RUNS}"
