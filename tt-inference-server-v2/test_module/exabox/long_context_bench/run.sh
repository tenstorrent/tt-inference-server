#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# long_context_bench/run.sh - validates ISL/OSL>64K hang and ISL>32K garbage
# scenarios via vllm bench serve against the deployed endpoint.

set -Eeuo pipefail

echo "==> Running long-context probe (ISL=${INPUT_LEN}, OSL=${OUTPUT_LEN}, conc=${MAX_CONCURRENCY})"
mkdir -p "$OUT"
RESULT_FILE="$OUT/bench_isl${INPUT_LEN}_osl${OUTPUT_LEN}_conc${MAX_CONCURRENCY}_${JOB_SUFFIX}.json"

export OPENAI_API_BASE="$TARGET"

set +e
timeout "${REQUEST_TIMEOUT_SEC}" vllm bench serve \
  --base-url "$OPENAI_API_BASE" \
  --model "$MODEL" \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --save-result \
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
  echo "TIMEOUT after ${REQUEST_TIMEOUT_SEC}s — likely hang at ISL=${INPUT_LEN} OSL=${OUTPUT_LEN}"
elif [ $RC -ne 0 ]; then
  echo "vllm bench serve exited with rc=$RC"
else
  echo "Probe finished cleanly."
fi

ls -la "$OUT/"
echo "==> Tests completed successfully."
