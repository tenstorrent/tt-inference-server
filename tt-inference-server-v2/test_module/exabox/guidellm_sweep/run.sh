#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# guidellm_sweep/run.sh - concurrency sweep at fixed ISL/OSL/turns via
# guidellm benchmark, one run per rate in $RATES.

set -Eeuo pipefail

echo "==> Running guidellm concurrency sweep over rates: ${RATES}"
mkdir -p "$OUT"

BACKEND_KWARGS="{\"api_key\":\"${OPENAI_API_KEY}\"}"
if [ -n "${DATASET_PATH}" ]; then
  DATA_SPEC="${DATASET_PATH}"
else
  DATA_SPEC="turns=${TURNS},prompt_tokens=${PROMPT_TOKENS},output_tokens=${OUTPUT_TOKENS}"
fi

REPORT_TYPE="guidellm_benchmark"
for RATE in $(echo "${RATES}" | tr ',' ' '); do
  RESULT_FILE="$OUT/sweep_rate${RATE}_isl${PROMPT_TOKENS}_osl${OUTPUT_TOKENS}_turns${TURNS}_${JOB_SUFFIX}.json"
  echo "==> guidellm rate=${RATE} -> ${RESULT_FILE}"

  set +e
  timeout "${REQUEST_TIMEOUT_SEC}" guidellm benchmark \
    --target "${TARGET}" \
    --model "${MODEL}" \
    --processor "${PROCESSOR}" \
    --profile concurrent \
    --rate "${RATE}" \
    --max-seconds "${MAX_SECONDS}" \
    --max-errors "${MAX_ERRORS}" \
    --warmup "${WARMUP}" \
    --request-type chat_completions \
    --data "${DATA_SPEC}" \
    --backend-kwargs "${BACKEND_KWARGS}" \
    --output-path "${RESULT_FILE}"
  RC=$?
  set -e

  if [ -f "${RESULT_FILE}" ]; then
    jq --arg report_type "$REPORT_TYPE" --argjson server_info "$SERVER_INFO" \
      '. + {report_type: $report_type, server_info: $server_info}' \
      "${RESULT_FILE}" > "${RESULT_FILE}.tmp" && mv "${RESULT_FILE}.tmp" "${RESULT_FILE}"
  fi

  if [ $RC -eq 124 ]; then
    echo "TIMEOUT after ${REQUEST_TIMEOUT_SEC}s at rate=${RATE} — continuing sweep"
  elif [ $RC -ne 0 ]; then
    echo "guidellm rate=${RATE} exited rc=${RC} — continuing sweep"
  else
    echo "rate=${RATE} done."
  fi
done

echo "Sweep complete."
ls -la "$OUT/"
echo "==> Tests completed successfully."
