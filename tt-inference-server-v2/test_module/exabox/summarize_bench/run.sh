#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# summarize_bench/run.sh - runs vllm bench serve against a custom prompt
# dataset (defaults to sibling prompts_32k.jsonl) for repeatable long-input
# summarization measurements.

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Resolve the dataset path: explicit env override > sibling default.
if [ -z "${DATASET_PATH}" ]; then
  DATASET_PATH="$SCRIPT_DIR/prompts_32k.jsonl"
fi
[ -f "$DATASET_PATH" ] || { echo "ERROR: dataset not found at $DATASET_PATH" >&2; exit 1; }

export OPENAI_API_BASE="$TARGET"

mkdir -p "$OUT"
RESULT_FILE="$OUT/${RESULT_NAME}_${JOB_SUFFIX}.json"

echo "==> Running summarize bench (dataset=${DATASET_PATH}, OSL=${OUTPUT_LEN}, conc=${MAX_CONCURRENCY})"

set +e
timeout "${REQUEST_TIMEOUT_SEC}" vllm bench serve \
  --base-url "$OPENAI_API_BASE" \
  --model "$MODEL" \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name custom \
  --dataset-path "$DATASET_PATH" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --custom-output-len "$OUTPUT_LEN" \
  --save-result \
  --save-detailed \
  --result-filename "$RESULT_FILE"
RC=$?
set -e

REPORT_TYPE="vllm_bench_serve"
if [ -f "$RESULT_FILE" ]; then
  jq --arg report_type "$REPORT_TYPE" --argjson server_info "$SERVER_INFO" \
    '. + {report_type: $report_type, server_info: $server_info}' \
    "$RESULT_FILE" > "${RESULT_FILE}.tmp" && mv "${RESULT_FILE}.tmp" "$RESULT_FILE"
fi

if [ $RC -eq 124 ]; then
  echo "TIMEOUT after ${REQUEST_TIMEOUT_SEC}s — likely hang at OSL=${OUTPUT_LEN}"
elif [ $RC -ne 0 ]; then
  echo "vllm bench serve exited with rc=$RC"
else
  echo "Bench finished cleanly."
fi

ls -la "$OUT/"
echo "==> Tests completed successfully."
