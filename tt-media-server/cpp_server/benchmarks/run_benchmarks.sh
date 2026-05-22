# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${OPENAI_API_BASE:-http://localhost:8000}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
BACKEND="openai-chat"
ENDPOINT="/v1/chat/completions"
DATASET="random"
NUM_PROMPTS="${NUM_PROMPTS:-1}"
RESULTS_DIR="${RESULTS_DIR:-bench_results}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
REPORT_TYPE="vllm_bench_serve"

mkdir -p "$RESULTS_DIR"

run_bench() {
    local isl=$1 osl=$2 concurrency=$3
    local suffix="${JOB_SUFFIX:-$TIMESTAMP}"
    local filename="${RESULTS_DIR}/bench_isl${isl}_osl${osl}_conc${concurrency}_${suffix}.json"

    echo "=== Running: ISL=${isl} OSL=${osl} max-concurrency=${concurrency} ==="
    vllm bench serve \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --endpoint "$ENDPOINT" \
        --dataset-name "$DATASET" \
        --random-input-len "$isl" \
        --random-output-len "$osl" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$concurrency" \
        --save-result \
        --result-filename "$filename"
    jq --argjson isl "$isl" --argjson osl "$osl" --arg report_type "$REPORT_TYPE" \
        '. + {input_seq_len: $isl, output_seq_len: $osl, report_type: $report_type}' \
        "$filename" > "${filename}.tmp" && mv "${filename}.tmp" "$filename"
    echo "  -> Saved to $filename"
}

# --- Phase 1: Increase ISL, fixed OSL=128, concurrency=64 ---
echo ""
echo "######################################"
echo "# Phase 1: Varying ISL (OSL=128, concurrency=64)"
echo "######################################"
FIXED_OSL=128
for isl in 128 256 512 1024 2048 4096 8192; do
    run_bench "$isl" "$FIXED_OSL" 64
done

# --- Phase 2: Increase OSL, fixed ISL=128, concurrency=64 ---
echo ""
echo "######################################"
echo "# Phase 2: Varying OSL (ISL=128, concurrency=64)"
echo "######################################"
FIXED_ISL=128
for osl in 128 256 512 1024 2048 4096 8192; do
    run_bench "$FIXED_ISL" "$osl" 64
done

# --- Phase 3: Increase concurrency, fixed ISL=512, OSL=512 ---
echo ""
echo "######################################"
echo "# Phase 3: Varying concurrency (ISL=512, OSL=512)"
echo "######################################"
FIXED_ISL=512
FIXED_OSL=512
for conc in 1 2 4 8 16 32 64; do
    run_bench "$FIXED_ISL" "$FIXED_OSL" "$conc"
done

echo ""
echo "All benchmarks complete. Results in ${RESULTS_DIR}/"
echo "Run: python plot_benchmarks.py ${RESULTS_DIR}/ to generate plots."

