#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Simple mode wrapper for a remote DeepSeek-R1-style server exposing
# OpenAI-compatible endpoints.
#
# Modes:
#   smoke  -> tiny end-to-end suite flow check
#   quick  -> 5-question AIME24 pass@1 x16 + small MMLU-Pro slice
#   single -> full AIME24
#   suite  -> AIME24 pass@1 x16 + GPQA Diamond + MATH-500 + LiveCodeBench + full MMLU-Pro

set -euo pipefail

usage() {
    cat <<'EOF' >&2
usage: run_deepseek_external.sh {smoke|quick|single|suite}

Environment knobs:
  OUTPUT_DIR             Root output directory for the selected mode
  SMOKE_LIMIT            Per-task limit for smoke mode (default: 2)
  SMOKE_AIME_RUNS        AIME samples per problem in smoke mode (default: 2)
  QUICK_MMLU_PRO_LIMIT   MMLU-Pro sample count for quick mode (default: 32)
  AIME_PASS1_RUNS        AIME samples per problem in quick/suite mode (default: 16)
  AIME_PASS1_PARALLEL_RUNS
                         Parallel AIME sample runs; quick defaults to 6 to keep ~30 AIME requests in flight

Notes:
  - Full MMLU-Pro is 12,032 questions, so suite mode is reporting-grade and long-running.
  - Use run_mmlu_pro_external.sh if you want to run just MMLU-Pro by itself.
EOF
    exit 64
}

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
    usage
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
RUNNER="${HERE}/helper_external_lm_eval.sh"
AIME_PASS1_RUNNER="${HERE}/run_aime24_pass1x16_external.sh"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

run_task() {
    local label="$1"
    shift
    echo "==> ${label}"
    "$@"
}

case "${MODE}" in
    smoke)
        ROOT_OUTPUT="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/deepseek_external_smoke_${TIMESTAMP}}"
        SMOKE_LIMIT="${SMOKE_LIMIT:-2}"
        SMOKE_AIME_RUNS="${SMOKE_AIME_RUNS:-2}"
        run_task "AIME24 short pass@1 smoke (${SMOKE_AIME_RUNS}x 5 questions)" \
            env \
            AIME_PASS1_TASK="r1_aime24_short" \
            AIME_PASS1_RUNS="${SMOKE_AIME_RUNS}" \
            AIME_PASS1_PARALLEL_RUNS="${AIME_PASS1_PARALLEL_RUNS:-${SMOKE_AIME_RUNS}}" \
            OUTPUT_DIR="${ROOT_OUTPUT}/r1_aime24_short_pass1x${SMOKE_AIME_RUNS}" \
            "${AIME_PASS1_RUNNER}"
        run_task "GPQA Diamond smoke (${SMOKE_LIMIT} questions)" \
            "${RUNNER}" \
            --task r1_gpqa_diamond \
            --chat-api \
            --num-fewshot 0 \
            --limit "${SMOKE_LIMIT}" \
            --output-dir "${ROOT_OUTPUT}/r1_gpqa_diamond_limit${SMOKE_LIMIT}"
        run_task "MATH-500 smoke (${SMOKE_LIMIT} questions)" \
            "${RUNNER}" \
            --task r1_math500 \
            --chat-api \
            --num-fewshot 0 \
            --limit "${SMOKE_LIMIT}" \
            --output-dir "${ROOT_OUTPUT}/r1_math500_limit${SMOKE_LIMIT}"
        run_task "LiveCodeBench smoke (${SMOKE_LIMIT} questions)" \
            "${RUNNER}" \
            --task livecodebench \
            --chat-api \
            --limit "${SMOKE_LIMIT}" \
            --output-dir "${ROOT_OUTPUT}/livecodebench_limit${SMOKE_LIMIT}"
        run_task "MMLU-Pro smoke (${SMOKE_LIMIT} questions/category)" \
            "${RUNNER}" \
            --task mmlu_pro \
            --chat-api \
            --limit "${SMOKE_LIMIT}" \
            --output-dir "${ROOT_OUTPUT}/mmlu_pro_limit${SMOKE_LIMIT}"
        ;;
    quick)
        ROOT_OUTPUT="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/deepseek_external_quick_${TIMESTAMP}}"
        QUICK_MMLU_PRO_LIMIT="${QUICK_MMLU_PRO_LIMIT:-32}"
        AIME_PASS1_RUNS="${AIME_PASS1_RUNS:-16}"
        AIME_PASS1_PARALLEL_RUNS="${AIME_PASS1_PARALLEL_RUNS:-6}"
        run_task "AIME24 short pass@1 (${AIME_PASS1_RUNS}x 5 questions)" \
            env \
            AIME_PASS1_TASK="r1_aime24_short" \
            AIME_PASS1_RUNS="${AIME_PASS1_RUNS}" \
            AIME_PASS1_PARALLEL_RUNS="${AIME_PASS1_PARALLEL_RUNS}" \
            OUTPUT_DIR="${ROOT_OUTPUT}/r1_aime24_short_pass1x${AIME_PASS1_RUNS}" \
            "${AIME_PASS1_RUNNER}"
        run_task "MMLU-Pro smoke (${QUICK_MMLU_PRO_LIMIT} questions)" \
            "${RUNNER}" \
            --task mmlu_pro \
            --chat-api \
            --limit "${QUICK_MMLU_PRO_LIMIT}" \
            --output-dir "${ROOT_OUTPUT}/mmlu_pro_limit${QUICK_MMLU_PRO_LIMIT}"
        ;;
    single)
        ROOT_OUTPUT="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/deepseek_external_single_${TIMESTAMP}}"
        exec "${RUNNER}" \
            --task r1_aime24 \
            --chat-api \
            --num-fewshot 0 \
            --output-dir "${ROOT_OUTPUT}/r1_aime24"
        ;;
    suite)
        ROOT_OUTPUT="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/deepseek_external_suite_${TIMESTAMP}}"
        AIME_PASS1_RUNS="${AIME_PASS1_RUNS:-16}"
        run_task "AIME24 pass@1 (${AIME_PASS1_RUNS}x full runs)" \
            env \
            AIME_PASS1_RUNS="${AIME_PASS1_RUNS}" \
            OUTPUT_DIR="${ROOT_OUTPUT}/r1_aime24_pass1x${AIME_PASS1_RUNS}" \
            "${AIME_PASS1_RUNNER}"
        run_task "GPQA Diamond reasoning (198 questions)" \
            "${RUNNER}" \
            --task r1_gpqa_diamond \
            --chat-api \
            --num-fewshot 0 \
            --output-dir "${ROOT_OUTPUT}/r1_gpqa_diamond"
        run_task "MATH-500 reasoning (500 questions)" \
            "${RUNNER}" \
            --task r1_math500 \
            --chat-api \
            --num-fewshot 0 \
            --output-dir "${ROOT_OUTPUT}/r1_math500"
        run_task "LiveCodeBench code_generation_lite" \
            "${RUNNER}" \
            --task livecodebench \
            --chat-api \
            --output-dir "${ROOT_OUTPUT}/livecodebench"
        run_task "MMLU-Pro full (12032 questions)" \
            "${RUNNER}" \
            --task mmlu_pro \
            --chat-api \
            --output-dir "${ROOT_OUTPUT}/mmlu_pro"
        ;;
    *)
        usage
        ;;
esac
