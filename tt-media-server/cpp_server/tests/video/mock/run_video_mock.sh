#!/usr/bin/env bash
# Compile the local video mock, restart it, and run the checks whose
# result is printed in this same terminal.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
# jsoncpp comes from the system package (libjsoncpp-dev)
if ! pkg-config --exists jsoncpp 2>/dev/null; then
  echo "jsoncpp not found: install libjsoncpp-dev (pkg-config must know jsoncpp)"
  exit 1
fi
JSON_FLAGS="$(pkg-config --cflags --libs jsoncpp)"
BIN=/tmp/video_mock_http
LOG=/tmp/video_mock_http.log
PORT=18080

echo "=== what the mock is ==="
echo "A local HTTP process at 127.0.0.1:${PORT}."
echo "It uses VideoGenerateRequest::fromJson and VideoJobStore, the same code as the service,"
echo "and the same job routes. There is no Drogon, worker, model or device."
echo "A job finishes at once, the result is mock://video/<task_id>, not an mp4 file."
echo

echo "=== compiling ==="
g++ -std=c++20 -O2 -pthread \
  -I"$ROOT/include" \
  "$HERE/video_mock_http.cpp" \
  $JSON_FLAGS \
  -o "$BIN"
echo "binary: $BIN"
echo

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
fi
if [[ -f /proc/net/tcp ]]; then
  : 
fi
# Stop a previous copy so this binary can bind the port.
for pid in $(ps -eo pid,cmd | awk '/\/tmp\/video_mock_http/ && !/awk/ {print $1}'); do
  kill "$pid" 2>/dev/null || true
done
sleep 0.2

: >"$LOG"
"$BIN" >>"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
sleep 0.3
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "the mock did not stay up. log:"
  cat "$LOG"
  exit 1
fi

shown=0
show_server() {
  local total
  total=$(wc -l <"$LOG" | tr -d ' ')
  if (( total > shown )); then
    echo "server:"
    tail -n +"$((shown + 1))" "$LOG" | sed 's/^/  /'
    shown=$total
  fi
}

run_case() {
  local title="$1"
  local checked="$2"
  shift 2
  echo
  echo "=== $title ==="
  echo "command: curl $(printf '%q ' "$@")"
  echo -n "response: "
  LAST_RESPONSE="$(curl -sS "$@" || echo "(curl failed)")"
  echo "$LAST_RESPONSE"
  echo
  show_server
  echo "checked: $checked"
}

show_server

run_case "1. liveness" \
  "the process listens and GET /tt-liveness returns model_ready=true and runner=video-mock." \
  -sS "http://127.0.0.1:${PORT}/tt-liveness"

run_case "2. valid prompt (creates a job)" \
  "prompt, seed and num_frames pass fromJson; the response is 202 with a job id and status=queued, without video_path." \
  -sS -X POST "http://127.0.0.1:${PORT}/v1/videos/generations" \
  -H 'content-type: application/json' \
  -d '{"prompt":"a mountain stream","seed":1,"num_frames":81}'

JOB_ID="$(printf '%s' "$LAST_RESPONSE" | python3 -c 'import sys, json; print(json.load(sys.stdin)["id"])')"

run_case "2b. polling the job" \
  "GET /v1/videos/generations/{id} returns the metadata; the mock runner has already finished the job (status=completed)." \
  -sS "http://127.0.0.1:${PORT}/v1/videos/generations/${JOB_ID}"

run_case "2c. job list" \
  "GET /v1/videos/jobs returns an array with that job." \
  -sS "http://127.0.0.1:${PORT}/v1/videos/jobs"

run_case "2d. deleting the job" \
  "DELETE of a finished job returns {id, object: video, deleted: true}." \
  -sS -X DELETE "http://127.0.0.1:${PORT}/v1/videos/generations/${JOB_ID}"

run_case "3. empty prompt" \
  "a prompt made only of spaces gets 422 and the message prompt must not be empty. The runner is not called." \
  -sS -X POST "http://127.0.0.1:${PORT}/v1/videos/generations" \
  -H 'content-type: application/json' \
  -d '{"prompt":"   "}'

run_case "4. num_frames is not positive" \
  "num_frames=0 gets 422 and the message num_frames must be positive." \
  -sS -X POST "http://127.0.0.1:${PORT}/v1/videos/generations" \
  -H 'content-type: application/json' \
  -d '{"prompt":"a mountain stream","num_frames":0}'

run_case "5. invalid JSON" \
  "a body that is not JSON gets 422 Invalid JSON body, before fromJson." \
  -sS -X POST "http://127.0.0.1:${PORT}/v1/videos/generations" \
  -H 'content-type: application/json' \
  -d 'this is not json'

run_case "6. unknown route" \
  "a route other than /tt-liveness and /v1/videos/generations gets 404." \
  -sS "http://127.0.0.1:${PORT}/no-such-route"

echo
echo "=== end ==="
echo "This checked the parser, the job flow (create, poll, list, delete) and the shape of the JSON responses."
echo "It did not check tt_media_server_cpp, the worker, media IPC or the device."
