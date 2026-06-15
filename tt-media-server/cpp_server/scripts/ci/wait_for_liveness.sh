#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

set -euo pipefail

HOST="127.0.0.1"
PORT=""
TIMEOUT_SECONDS=30
INTERVAL_SECONDS=1
EXPECT='"model_ready":true'
PID_FILE=""
LABEL="server"

usage() {
    cat <<'EOF'
Usage: wait_for_liveness.sh --port PORT [options]

Options:
  --host HOST             Host to poll (default: 127.0.0.1)
  --port PORT             HTTP port to poll (required)
  --timeout SECONDS       Total wait time (default: 30)
  --interval SECONDS      Poll interval (default: 1)
  --expect STRING         String expected in /tt-liveness response
                           (default: "model_ready":true)
  --pid-file PATH         Fail early if this process exits
  --label LABEL           Friendly name for logs/errors
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
        --interval) INTERVAL_SECONDS="$2"; shift 2 ;;
        --expect) EXPECT="$2"; shift 2 ;;
        --pid-file) PID_FILE="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$PORT" ]]; then
    echo "--port is required" >&2
    usage >&2
    exit 2
fi

deadline=$((SECONDS + TIMEOUT_SECONDS))
last_response=""

while (( SECONDS < deadline )); do
    if [[ -n "$PID_FILE" && -f "$PID_FILE" ]]; then
        pid="$(cat "$PID_FILE")"
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "::error::${LABEL} process ${pid} exited before becoming ready"
            exit 1
        fi
    fi

    last_response="$(curl -sf "http://${HOST}:${PORT}/tt-liveness" 2>/dev/null || true)"
    if [[ -n "$last_response" ]] && grep -Fq "$EXPECT" <<<"$last_response"; then
        echo "${LABEL} ready on ${HOST}:${PORT}."
        exit 0
    fi
    sleep "$INTERVAL_SECONDS"
done

echo "::error::${LABEL} did not become ready on ${HOST}:${PORT} within ${TIMEOUT_SECONDS}s"
if [[ -n "$last_response" ]]; then
    echo "Last /tt-liveness response:"
    printf '%s\n' "$last_response"
fi
exit 1
