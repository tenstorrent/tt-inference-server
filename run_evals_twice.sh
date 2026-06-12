#!/usr/bin/env bash
# Run terminal_bench_2 evals twice, preserving results from each trial.
# Usage:
#   HF_TOKEN="..." API_KEY="..." ./run_evals_twice.sh
set -euo pipefail

AGENTIC_DIR="workflow_logs/evals_output/eval_id_tt-transformers_Kimi-K2.6_gpu/agentic"
TASK_DIR="${AGENTIC_DIR}/terminal_bench_2"
EVAL_CMD=".venv/bin/python run.py --model Kimi-K2.6 --workflow evals --device gpu --server-url https://console.tenstorrent.com:443"

cd "$(dirname "$0")"

run_trial() {
    local trial=$1
    local dest="${AGENTIC_DIR}/terminal_bench_2_trial${trial}"

    echo "========================================="
    echo "  Starting trial ${trial}"
    echo "========================================="

    if [ -d "${dest}" ]; then
        echo "ERROR: destination already exists: ${dest}"
        echo "Delete or rename it manually before re-running."
        exit 1
    fi

    ${EVAL_CMD} || true

    if [ -d "${TASK_DIR}" ]; then
        echo "Saving trial ${trial} results: ${TASK_DIR} -> ${dest}"
        mv "${TASK_DIR}" "${dest}" || {
            echo "ERROR: mv failed — results are still safe in ${TASK_DIR}"
            exit 1
        }
    else
        echo "WARNING: expected output dir not found: ${TASK_DIR}"
    fi
}

run_trial 1
run_trial 2

echo "========================================="
echo "  Both trials complete."
echo "  Trial 1: ${AGENTIC_DIR}/terminal_bench_2_trial1"
echo "  Trial 2: ${AGENTIC_DIR}/terminal_bench_2_trial2"
echo "========================================="
