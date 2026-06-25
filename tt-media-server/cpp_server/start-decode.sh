#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# Launch a tt_media_server_cpp in disaggregated *decode* mode.
#
# The decode server is the one clients talk to (send your chat requests here).
# It binds the inter-server socket (as the socket SERVER) that the prefill
# server connects to, and routes prefill work across that socket.
#
# MAX_TOKENS_TO_PREFILL_ON_DECODE=0 forces *every* request across the socket to
# the prefill server (true disaggregation). Raise it to prefill short prompts
# locally on the decode server instead.
#
# Usage (flags accepted in any position):
#   ./start-decode.sh                  # HTTP :8000, socket bind :9000
#   ./start-decode.sh -p 8000          # override HTTP port (chat.sh defaults to 8000)
#   ./start-decode.sh -s 9000 -T tcp   # inter-server port / transport
#
# Distinct shared-memory queue names (the *_decode suffix) keep this process
# from colliding with a prefill server on the same machine — sharing them is
# what makes disaggregation hang.

set -euo pipefail

HTTP_PORT=8000                 # OpenAI HTTP port clients hit (chat.sh default)
SOCKET_HOST=127.0.0.1          # inter-server socket bind host
SOCKET_PORT=9000               # inter-server socket bind port
SOCKET_TRANSPORT=tcp           # tcp or zmq (must match the prefill server)
MAX_PREFILL_ON_DECODE=0        # 0 = always offload prefill to prefill server
THREADS=4
LOG_LEVEL=debug

usage() {
  cat >&2 <<EOF
Usage: $0 [-p HTTP_PORT] [-H SOCKET_HOST] [-s SOCKET_PORT] [-T tcp|zmq] [-m MAX_TOKENS] [-t THREADS]
  -p HTTP_PORT      OpenAI HTTP listen port            (default: $HTTP_PORT)
  -H SOCKET_HOST    inter-server bind host              (default: $SOCKET_HOST)
  -s SOCKET_PORT    inter-server bind port              (default: $SOCKET_PORT)
  -T TRANSPORT      socket transport tcp|zmq             (default: $SOCKET_TRANSPORT)
  -m MAX_TOKENS     MAX_TOKENS_TO_PREFILL_ON_DECODE      (default: $MAX_PREFILL_ON_DECODE)
  -t THREADS        worker threads (-t)                  (default: $THREADS)
EOF
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    -p) HTTP_PORT="$2"; shift 2 ;;
    -H) SOCKET_HOST="$2"; shift 2 ;;
    -s) SOCKET_PORT="$2"; shift 2 ;;
    -T) SOCKET_TRANSPORT="$2"; shift 2 ;;
    -m) MAX_PREFILL_ON_DECODE="$2"; shift 2 ;;
    -t) THREADS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/build/tt_media_server_cpp"
if [ ! -x "$BIN" ]; then
  echo "[start-decode] binary not found at $BIN — build it first." >&2
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG="$SCRIPT_DIR/decode-${TS}.log"

echo "[start-decode] HTTP :$HTTP_PORT  socket=$SOCKET_TRANSPORT://$SOCKET_HOST:$SOCKET_PORT  max_prefill_on_decode=$MAX_PREFILL_ON_DECODE  log=$LOG" >&2

SOCKET_TRANSPORT="$SOCKET_TRANSPORT" \
LLM_MODE=decode \
MAX_TOKENS_TO_PREFILL_ON_DECODE="$MAX_PREFILL_ON_DECODE" \
SOCKET_HOST="$SOCKET_HOST" \
SOCKET_PORT="$SOCKET_PORT" \
TT_WARMUP_SIGNALS_QUEUE=tt_warmup_signals_decode \
TT_TASK_QUEUE=tt_tasks_decode \
TT_RESULT_QUEUE=tt_results_decode \
TT_CANCEL_QUEUE=tt_cancels_decode \
TT_MEMORY_REQUEST_QUEUE=tt_mem_requests_decode \
TT_MEMORY_RESULT_QUEUE=tt_mem_results_decode \
TT_WORKER_METRICS_SHM=/tt_worker_metrics_decode \
TT_LOG_LEVEL="$LOG_LEVEL" \
"$BIN" -p "$HTTP_PORT" -t "$THREADS" |& tee "$LOG"
