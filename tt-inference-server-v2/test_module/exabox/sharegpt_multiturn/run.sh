#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# sharegpt_multiturn/run.sh - runs guidellm against /v1/chat/completions using
# real ShareGPT human prompts; assistant turns come from the server under test.

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

mkdir -p "$OUT"

echo "==> Running ShareGPT multi-turn benchmark"
"$SCRIPT_DIR/run_sharegpt_multiturn.sh" \
  --target "$TARGET" \
  --model "$MODEL" \
  --api-key "$OPENAI_API_KEY" \
  --concurrency "$CONCURRENCY" \
  --duration "$DURATION" \
  --min-turns "$MIN_TURNS" \
  --max-turns "$MAX_TURNS" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --output-dir "$OUT"

echo "==> ShareGPT multi-turn benchmark complete; annotating results"
REPORT_TYPE="guidellm_benchmark"
shopt -s nullglob
for f in "$OUT"/benchmark_*.json; do
  jq empty "$f" 2>/dev/null || continue
  jq --arg report_type "$REPORT_TYPE" --argjson server_info "$SERVER_INFO" \
    '. + {report_type: $report_type, server_info: $server_info}' \
    "$f" > "${f}.tmp" \
    && mv "${f}.tmp" "$OUT/sharegpt_multiturn_${JOB_SUFFIX}.json"
  [ "$f" != "$OUT/sharegpt_multiturn_${JOB_SUFFIX}.json" ] && rm -f "$f"
done

ls -la "$OUT/"
echo "==> Tests completed successfully."
