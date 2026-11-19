#!/usr/bin/env bash
# HTTP contract for the video service. With VIDEO_BASE_URL set, talks to that
# server. Without it, builds and starts the local stub on 127.0.0.1:18080.
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
PORT=18080

if [[ -n "${VIDEO_BASE_URL:-}" ]]; then
  exec python3 "$HERE/contract_cases.py" --base-url "$VIDEO_BASE_URL"
fi

echo "VIDEO_BASE_URL is not set, using the local stub at 127.0.0.1:${PORT}"
g++ -std=c++20 -O2 -pthread \
  -I"$ROOT/include" \
  "$HERE/video_mock_http.cpp" \
  $JSON_FLAGS \
  -o "$BIN"

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
fi
for pid in $(ps -eo pid,cmd | awk '/\/tmp\/video_mock_http/ && !/awk/ {print $1}'); do
  kill "$pid" 2>/dev/null || true
done
sleep 0.2

"$BIN" >/tmp/video_mock_http.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
sleep 0.3
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "the stub did not stay up"
  cat /tmp/video_mock_http.log
  exit 1
fi

python3 "$HERE/contract_cases.py" --base-url "http://127.0.0.1:${PORT}"
