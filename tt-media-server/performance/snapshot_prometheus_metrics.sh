#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Snapshot Prometheus metrics relevant to video inference + post-processing.
# Usage:
#   ENABLE_TELEMETRY=true ./performance/snapshot_prometheus_metrics.sh [output.txt]
# Env:
#   METRICS_URL  (default http://127.0.0.1:${SERVICE_PORT:-8000}/metrics)

set -euo pipefail

PORT="${SERVICE_PORT:-8000}"
URL="${METRICS_URL:-http://127.0.0.1:${PORT}/metrics}"
OUT="${1:-metrics-snapshot-$(date +%Y%m%d-%H%M%S).txt}"

# Match HELP/TYPE for our namespaces and all sample lines (buckets, sum, count, gauges).
# Do not pipe curl -> grep in one pipeline: with set -e, grep exits 1 when there are no
# matches and the whole script fails (and some IDE terminals close on non-zero exit).
metrics_body="$(mktemp)"
trap "rm -f \"${metrics_body}\"" EXIT

curl -sS --fail "$URL" -o "$metrics_body"
grep -E \
  '^(# HELP|# TYPE).*tt_(media_server|video_pipeline)|^tt_(media_server|video_pipeline)_' \
  <"$metrics_body" >"$OUT" || true

echo "Wrote ${OUT} ($(wc -l <"$OUT") lines)"
