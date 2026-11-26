#!/usr/bin/env bash
# Server-overhead benchmark with the mock video runner (no device): starts
# tt_media_server_cpp with a zero-delay mock runner and runs the `throughput` and
# `control` phases of video_benchmark.py. Needs a built ./build/tt_media_server_cpp.
# Usage (from cpp_server): ./tests/video/benchmark/run_mock_benchmark.sh [out_dir]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$(mktemp -d /tmp/tt-video-bench.XXXXXX)}"
PORT="${PORT:-18111}"
KEY=test
mkdir -p "$OUT"
env MODEL_SERVICE=video MODEL_RUNNER_TYPE=video_mock MODEL_RUNNER=tt-wan2.2 \
  OPENAI_API_KEY="$KEY" TT_VIDEO_OUTPUT_DIR="$OUT/videos" \
  TT_MEDIA_TASK_QUEUE=bm_tasks TT_MEDIA_RESULT_QUEUE=bm_results \
  TT_WARMUP_SIGNALS_QUEUE=bm_warm TT_WORKER_METRICS_SHM=/bm_metrics \
  TT_VIDEO_MOCK_DELAY_MS="${MOCK_DELAY_MS:-0}" TT_VIDEO_MOCK_WRITE_FILE=1 \
  "$ROOT/build/tt_media_server_cpp" -p "$PORT" >"$OUT/server.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
ready=0
for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:$PORT/tt-liveness" | grep -q '"model_ready":true'; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  echo "the server did not become ready (log: $OUT/server.log)" >&2
  exit 1
fi
BASE="http://127.0.0.1:$PORT"
python3 "$HERE/video_benchmark.py" --phase throughput --base-url "$BASE" --api-key "$KEY" \
  --jobs "${JOBS:-300}" --clients "${CLIENTS:-8}" --steps 4 --poll 0.05 --out "$OUT/throughput.json"
python3 "$HERE/video_benchmark.py" --phase control --base-url "$BASE" --api-key "$KEY" \
  --steps 4 --out "$OUT/control.json"
curl -sf "$BASE/metrics" | grep '^tt_video' >"$OUT/metrics_tt_video.txt"
echo "results in $OUT"
