#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

set -euo pipefail

COMMAND=""
TIMEOUT_SECONDS=30
INTERVAL_SECONDS=1
PID_FILE=""
LABEL="command"
FAILURE_LOG=""

usage() {
    cat <<'EOF'
Usage: wait_for_command.sh --command COMMAND [options]

Options:
  --command COMMAND       Shell command to retry until it succeeds (required)
  --timeout SECONDS       Total wait time (default: 30)
  --interval SECONDS      Poll interval (default: 1)
  --pid-file PATH         Fail early if this process exits
  --failure-log PATH      Print this log if the wait times out
  --label LABEL           Friendly name for logs/errors
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --command) COMMAND="$2"; shift 2 ;;
        --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
        --interval) INTERVAL_SECONDS="$2"; shift 2 ;;
        --pid-file) PID_FILE="$2"; shift 2 ;;
        --failure-log) FAILURE_LOG="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$COMMAND" ]]; then
    echo "--command is required" >&2
    usage >&2
    exit 2
fi

deadline=$((SECONDS + TIMEOUT_SECONDS))

while (( SECONDS < deadline )); do
    if [[ -n "$PID_FILE" && -f "$PID_FILE" ]]; then
        pid="$(cat "$PID_FILE")"
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "::error::${LABEL} process ${pid} exited before ${LABEL} became ready"
            [[ -n "$FAILURE_LOG" && -f "$FAILURE_LOG" ]] && tail -100 "$FAILURE_LOG"
            exit 1
        fi
    fi

    if bash -c "$COMMAND"; then
        echo "${LABEL} ready."
        exit 0
    fi
    sleep "$INTERVAL_SECONDS"
done

echo "::error::${LABEL} did not become ready within ${TIMEOUT_SECONDS}s"
if [[ -n "$FAILURE_LOG" && -f "$FAILURE_LOG" ]]; then
    echo "Last 100 lines of ${FAILURE_LOG}:"
    tail -100 "$FAILURE_LOG"
fi
exit 1
