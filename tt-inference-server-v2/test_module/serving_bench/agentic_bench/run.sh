#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# agentic_bench/run.sh - runs the agentic-shape guidellm soak against the
# deployed endpoint, then stamps report_type + server_info onto the result.

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

echo "==> Running agentic benchmark"
"$SCRIPT_DIR/agentic_bench.sh"

echo "==> Annotating results"
REPORT_TYPE="guidellm_benchmark"
shopt -s nullglob
for f in "$OUT"/*.json; do
  jq empty "$f" 2>/dev/null || continue
  jq --arg report_type "$REPORT_TYPE" --argjson server_info "$SERVER_INFO" \
    '. + {report_type: $report_type, server_info: $server_info}' \
    "$f" > "${f}.tmp" \
    && mv "${f}.tmp" "$OUT/benchmarks_${JOB_SUFFIX}.json"
  # If the source name differed from the new name, remove the original.
  [ "$f" != "$OUT/benchmarks_${JOB_SUFFIX}.json" ] && rm -f "$f"
done

ls -la "$OUT/"
echo "==> Tests completed successfully."
