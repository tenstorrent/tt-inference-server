#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# Launch a tt_media_server_cpp in disaggregated *prefill* mode.
#
# The prefill server connects (as the socket CLIENT) to the decode server's
# inter-server socket and serves prefill requests routed to it from decode.
# Start the decode server (./start-decode.sh) first, or start this one — they
# retry the connection, so order is not strict.
#
# Usage (flags accepted in any position):
#   ./start-prefill.sh                       # HTTP :8001, socket -> 127.0.0.1:9000
#   ./start-prefill.sh -p 8001               # override HTTP port
#   ./start-prefill.sh -H 10.0.0.5 -s 9000   # decode host/port to connect to
#
# Distinct shared-memory queue names (the *_prefill suffix) keep this process
# from colliding with a decode server on the same machine — sharing them is
# what makes disaggregation hang.

set -euo pipefail

HTTP_PORT=8001                 # this server's OpenAI HTTP port
SOCKET_HOST=127.0.0.1          # decode server's inter-server socket host
SOCKET_PORT=9000               # decode server's inter-server socket port
SOCKET_TRANSPORT=tcp           # tcp or zmq (must match the decode server)
DEVICE_BACKEND=mock_pipeline   # prefill backend
THREADS=4
LOG_LEVEL=debug

usage() {
  cat >&2 <<EOF
Usage: $0 [-p HTTP_PORT] [-H SOCKET_HOST] [-s SOCKET_PORT] [-T tcp|zmq] [-b BACKEND] [-t THREADS]
  -p HTTP_PORT      OpenAI HTTP listen port      (default: $HTTP_PORT)
  -H SOCKET_HOST    decode inter-server host      (default: $SOCKET_HOST)
  -s SOCKET_PORT    decode inter-server port      (default: $SOCKET_PORT)
  -T TRANSPORT      socket transport tcp|zmq       (default: $SOCKET_TRANSPORT)
  -b BACKEND        LLM_DEVICE_BACKEND             (default: $DEVICE_BACKEND)
  -t THREADS        worker threads (-t)            (default: $THREADS)
EOF
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    -p) HTTP_PORT="$2"; shift 2 ;;
    -H) SOCKET_HOST="$2"; shift 2 ;;
    -s) SOCKET_PORT="$2"; shift 2 ;;
    -T) SOCKET_TRANSPORT="$2"; shift 2 ;;
    -b) DEVICE_BACKEND="$2"; shift 2 ;;
    -t) THREADS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/build/tt_media_server_cpp"
if [ ! -x "$BIN" ]; then
  echo "[start-prefill] binary not found at $BIN — build it first." >&2
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="$SCRIPT_DIR/prefill-${TS}.log"

echo "[start-prefill] HTTP :$HTTP_PORT  socket=$SOCKET_TRANSPORT://$SOCKET_HOST:$SOCKET_PORT  backend=$DEVICE_BACKEND  log=$LOG" >&2

SOCKET_TRANSPORT="$SOCKET_TRANSPORT" \
LLM_MODE=prefill \
LLM_DEVICE_BACKEND="$DEVICE_BACKEND" \
SOCKET_HOST="$SOCKET_HOST" \
SOCKET_PORT="$SOCKET_PORT" \
TT_WARMUP_SIGNALS_QUEUE=tt_warmup_signals_prefill \
TT_TASK_QUEUE=tt_tasks_prefill \
TT_RESULT_QUEUE=tt_results_prefill \
TT_CANCEL_QUEUE=tt_cancels_prefill \
TT_MEMORY_REQUEST_QUEUE=tt_mem_requests_prefill \
TT_MEMORY_RESULT_QUEUE=tt_mem_results_prefill \
TT_WORKER_METRICS_SHM=/tt_worker_metrics_prefill \
TT_LOG_LEVEL="$LOG_LEVEL" \
"$BIN" -p "$HTTP_PORT" -t "$THREADS" |& tee "$LOG"
