#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# benchmark/run.sh - invokes run_benchmarks.sh against the deployed endpoint.
# $INFERENCE_SERVER_DIR defaults to this repo's root (set by the dispatcher);
# pass --inference-server-dir to run against a different checkout.
#
# Env (from dispatcher / defaults.env):
#   TARGET                    http://<helm-release>-server:8000
#   OPENAI_API_KEY            bearer token for the server
#   INFERENCE_SERVER_DIR      path to a checked-out tt-inference-server
#   OUT                       output dir for result JSON
#   JOB_SUFFIX                used by the upstream script for filenames
#   SERVER_INFO               JSON from /info, stamped onto results

set -Eeuo pipefail

: "${INFERENCE_SERVER_DIR:?INFERENCE_SERVER_DIR is required for the 'benchmark' test (path to a checked-out tt-inference-server)}"

BENCH_SCRIPT="$INFERENCE_SERVER_DIR/tt-media-server/cpp_server/benchmarks/run_benchmarks.sh"
[ -x "$BENCH_SCRIPT" ] || { echo "ERROR: $BENCH_SCRIPT not found or not executable" >&2; exit 1; }

export OPENAI_API_BASE="$TARGET"
export RESULTS_DIR="$OUT"

echo "==> Invoking $BENCH_SCRIPT"
cd "$INFERENCE_SERVER_DIR/tt-media-server"
"$BENCH_SCRIPT"

# Stamp server_info onto this attempt's results without reading shared outputs.
shopt -s nullglob
for f in "$RESULTS_DIR"/*.json; do
  [ -f "$f" ] || continue
  jq empty "$f" 2>/dev/null || { echo "warning: skipping non-JSON $f"; continue; }
  jq --argjson server_info "$SERVER_INFO" \
    '. + {server_info: $server_info}' \
    "$f" > "${f}.tmp"
  mv "${f}.tmp" "$f"
done

ls -la "$OUT"
echo "==> Tests completed successfully."
