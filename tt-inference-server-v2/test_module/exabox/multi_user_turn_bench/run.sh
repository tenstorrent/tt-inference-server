#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# multi_user_turn_bench/run.sh - runs N users x M turns via httpx streaming
# against /v1/chat/completions and writes a per-user log + statistics.

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

mkdir -p "$OUT"

FAST_MODE_FLAG=""
if [ "${FAST_MODE}" != "true" ]; then
  FAST_MODE_FLAG="--no-fast-mode"
fi

echo "==> Running multi-user multi-turn benchmark"
python3 "$SCRIPT_DIR/main.py" \
  --url "${TARGET}/v1/chat/completions" \
  --prompts "$SCRIPT_DIR/prompts.json" \
  --batch-size "${BATCH_SIZE}" \
  --max-tokens "${MAX_TOKENS}" \
  --timeout "${TIMEOUT}" \
  --log-file "$OUT/inference_log.txt" \
  $FAST_MODE_FLAG

echo "==> Multi-user multi-turn benchmark complete."
ls -la "$OUT/"
echo "==> Tests completed successfully."
