#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

set -euo pipefail

HOST="127.0.0.1"
TIMEOUT_SECONDS=30
INTERVAL_SECONDS=1
LABEL="port"
PORTS=()

usage() {
    cat <<'EOF'
Usage: wait_for_port_free.sh [options] PORT [PORT...]

Options:
  --host HOST             Host to check (default: 127.0.0.1)
  --port PORT             Port to check. May be repeated.
  --timeout SECONDS       Total wait time (default: 30)
  --interval SECONDS      Poll interval (default: 1)
  --label LABEL           Friendly name for logs/errors
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORTS+=("$2"); shift 2 ;;
        --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
        --interval) INTERVAL_SECONDS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) PORTS+=("$1"); shift ;;
    esac
done

if [[ "${#PORTS[@]}" -eq 0 ]]; then
    echo "At least one port is required" >&2
    usage >&2
    exit 2
fi

port_is_free() {
    local port="$1"
    python3 - "$HOST" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    sys.exit(1 if sock.connect_ex((host, port)) == 0 else 0)
PY
}

deadline=$((SECONDS + TIMEOUT_SECONDS))
busy=()

while (( SECONDS < deadline )); do
    busy=()
    for port in "${PORTS[@]}"; do
        if ! port_is_free "$port"; then
            busy+=("$port")
        fi
    done

    if [[ "${#busy[@]}" -eq 0 ]]; then
        echo "${LABEL} port(s) free on ${HOST}: ${PORTS[*]}."
        exit 0
    fi

    sleep "$INTERVAL_SECONDS"
done

echo "::error::${LABEL} port(s) still in use on ${HOST} after ${TIMEOUT_SECONDS}s: ${busy[*]}"
exit 1
