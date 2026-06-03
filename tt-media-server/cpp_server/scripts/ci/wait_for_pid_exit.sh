#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# Wait for a previously-started server PID to FULLY exit before the caller
# starts the next server. A freed TCP port is not enough: the HTTP socket
# closes early in shutdown ("Server shutdown complete") while the worker
# subprocess is still tearing down — and that teardown removes the
# Boost.Interprocess queues (e.g. the default-named tt_warmup_signals) from
# /dev/shm. When two servers share default queue names, starting the next one
# before this teardown finishes lets the dying worker delete the freshly
# created queue out from under the new worker ("Failed to open existing queue:
# tt_warmup_signals ... No such file or directory"). Waiting on the parent PID
# (which exits only after "LLMService Stopped") closes that window.

set -euo pipefail

TIMEOUT_SECONDS=30
INTERVAL_SECONDS=1
SIGKILL_AFTER_SECONDS=15
LABEL="server"
PID=""
PID_FILE=""

usage() {
    cat <<'EOF'
Usage: wait_for_pid_exit.sh (--pid PID | --pid-file PATH) [options]

Options:
  --pid PID               Process id to wait on
  --pid-file PATH         File containing the process id to wait on
  --timeout SECONDS       Total wait time before giving up (default: 30)
  --interval SECONDS      Poll interval (default: 1)
  --sigkill-after SECONDS Escalate from SIGTERM to SIGKILL after this long
                          (default: 15)
  --label LABEL           Friendly name for logs/errors
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid) PID="$2"; shift 2 ;;
        --pid-file) PID_FILE="$2"; shift 2 ;;
        --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
        --interval) INTERVAL_SECONDS="$2"; shift 2 ;;
        --sigkill-after) SIGKILL_AFTER_SECONDS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$PID" && -n "$PID_FILE" ]]; then
    [[ -f "$PID_FILE" ]] && PID="$(cat "$PID_FILE")"
fi

if [[ -z "$PID" ]]; then
    # No PID to wait on (server never started, or pid file missing). Nothing
    # to guard against — treat as already-exited so callers stay idempotent.
    echo "${LABEL}: no pid to wait on; assuming already exited."
    exit 0
fi

if ! kill -0 "$PID" 2>/dev/null; then
    echo "${LABEL} (pid ${PID}) already exited."
    exit 0
fi

deadline=$((SECONDS + TIMEOUT_SECONDS))
escalate_at=$((SECONDS + SIGKILL_AFTER_SECONDS))
escalated=0

while (( SECONDS < deadline )); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "${LABEL} (pid ${PID}) exited."
        exit 0
    fi
    if (( escalated == 0 && SECONDS >= escalate_at )); then
        echo "${LABEL} (pid ${PID}) still alive after ${SIGKILL_AFTER_SECONDS}s; sending SIGKILL."
        kill -9 "$PID" 2>/dev/null || true
        escalated=1
    fi
    sleep "$INTERVAL_SECONDS"
done

echo "::error::${LABEL} (pid ${PID}) did not exit within ${TIMEOUT_SECONDS}s"
exit 1
