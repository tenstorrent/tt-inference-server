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
AIME_PASS1_PARALLEL_RUNS="${AIME_PASS1_PARALLEL_RUNS:-1}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_OUTPUT="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/${TASK}_pass1x${RUNS}_${TIMESTAMP}}"
BASE_MODEL_SEED="${MODEL_SEED:-1234000}"

if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || [[ "${RUNS}" -lt 1 ]]; then
    echo "AIME_PASS1_RUNS must be a positive integer, got: ${RUNS}" >&2
    exit 64
fi
if ! [[ "${AIME_PASS1_PARALLEL_RUNS}" =~ ^[0-9]+$ ]] || [[ "${AIME_PASS1_PARALLEL_RUNS}" -lt 1 ]]; then
    echo "AIME_PASS1_PARALLEL_RUNS must be a positive integer, got: ${AIME_PASS1_PARALLEL_RUNS}" >&2
    exit 64
fi

mkdir -p "${ROOT_OUTPUT}"

include_args=()
if [[ "${TASK}" == "r1_aime24_short" ]]; then
    include_args=("${REPO_ROOT}/evals/custom_tasks/r1_aime24_short")
fi

echo "Running ${TASK} pass@1 estimate"
echo "Samples per problem: ${RUNS}"
echo "Parallel sample runs: ${AIME_PASS1_PARALLEL_RUNS}"
echo "Output dir: ${ROOT_OUTPUT}"

run_one() {
    local run_idx="$1"
    local run_label
    local run_output
    local run_seed

    run_label="$(printf 'sample_%02d' "${run_idx}")"
    run_output="${ROOT_OUTPUT}/${run_label}"
    run_seed=$((BASE_MODEL_SEED + run_idx - 1))

    echo
    echo "==> ${run_label}/${RUNS} (MODEL_SEED=${run_seed})"

    MODEL_SEED="${run_seed}" \
    OUTPUT_DIR="${run_output}" \
        "${RUNNER}" "${TASK}" "${include_args[@]}"
}

run_one_quiet() {
    local run_idx="$1"
    local run_label
    local run_output
    local run_seed
    local log_path

    run_label="$(printf 'sample_%02d' "${run_idx}")"
    run_output="${ROOT_OUTPUT}/${run_label}"
    run_seed=$((BASE_MODEL_SEED + run_idx - 1))
    log_path="${run_output}/lm_eval.log"

    mkdir -p "${run_output}"
    echo "==> starting ${run_label}/${RUNS} (MODEL_SEED=${run_seed}, log: ${log_path})"

    MODEL_SEED="${run_seed}" \
    OUTPUT_DIR="${run_output}" \
        "${RUNNER}" "${TASK}" "${include_args[@]}" > "${log_path}" 2>&1
}

if [[ "${AIME_PASS1_PARALLEL_RUNS}" -eq 1 ]]; then
    for run_idx in $(seq 1 "${RUNS}"); do
        run_one "${run_idx}"
    done
else
    next_run=1
    while [[ "${next_run}" -le "${RUNS}" ]]; do
        batch_pids=()
        batch_labels=()
        batch_logs=()
        batch_count=0

        while [[ "${next_run}" -le "${RUNS}" && "${batch_count}" -lt "${AIME_PASS1_PARALLEL_RUNS}" ]]; do
            run_label="$(printf 'sample_%02d' "${next_run}")"
            run_output="${ROOT_OUTPUT}/${run_label}"
            log_path="${run_output}/lm_eval.log"

            run_one_quiet "${next_run}" &
            batch_pids+=("$!")
            batch_labels+=("${run_label}")
            batch_logs+=("${log_path}")

            next_run=$((next_run + 1))
            batch_count=$((batch_count + 1))
        done

        batch_failures=0
        for index in "${!batch_pids[@]}"; do
            if wait "${batch_pids[$index]}"; then
                echo "==> completed ${batch_labels[$index]}/${RUNS}"
            else
                batch_failures=$((batch_failures + 1))
                echo "==> failed ${batch_labels[$index]}/${RUNS}; last log lines:" >&2
                tail -40 "${batch_logs[$index]}" >&2 || true
            fi
        done

        if [[ "${batch_failures}" -gt 0 ]]; then
            exit 1
        fi
    done
fi

echo
"${AGGREGATOR}" "${ROOT_OUTPUT}" --task "${TASK}" --expected-runs "${RUNS}"
