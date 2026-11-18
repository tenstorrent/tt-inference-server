#!/usr/bin/env bash
# Starts tt_media_server_cpp with the mock video runner (private shm names, no
# device) once per phase and runs tests/video/mock/server_cases.py against it.
# Needs a built ./build/tt_media_server_cpp. Usage (from cpp_server):
#   ./tests/video/mock/run_server_tests.sh
# Phases: contract (contract_cases.py), then main, sync-error, limits, warmup and
# cancel-race (server_cases.py).
# With one DEVICE_IDS group the runner runs on a thread of the server process
# (the default); TT_VIDEO_IN_PROCESS=0 ./tests/video/mock/run_server_tests.sh runs
# the same phases against a separate worker process.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-18099}"
KEY="${VIDEO_API_KEY:-test}"
OUT="$(mktemp -d /tmp/tt-video-server-tests.XXXXXX)"
BASE="http://127.0.0.1:$PORT"
export VIDEO_API_KEY="$KEY"
rc=0
PID=""

stop_server() {
  if [[ -n "$PID" ]]; then
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
    PID=""
  fi
}
trap 'stop_server; rm -rf "$OUT"' EXIT

# start_server <log> [ENV=VALUE ...]
start_server() {
  local log="$1"; shift
  env MODEL_SERVICE=video MODEL_RUNNER_TYPE=video_mock MODEL_RUNNER=tt-wan2.2 \
    OPENAI_API_KEY="$KEY" TT_VIDEO_OUTPUT_DIR="$OUT/videos" \
    TT_MEDIA_TASK_QUEUE=vst_tasks TT_MEDIA_RESULT_QUEUE=vst_results \
    TT_WARMUP_SIGNALS_QUEUE=vst_warm TT_WORKER_METRICS_SHM=/vst_metrics \
    "$@" "$ROOT/build/tt_media_server_cpp" -p "$PORT" >"$log" 2>&1 &
  PID=$!
}

wait_ready() {
  for _ in $(seq 1 120); do
    if curl -sf "$BASE/tt-liveness" | grep -q '"model_ready":true'; then return 0; fi
    sleep 1
  done
  return 1
}

run_phase() {  # run_phase <phase> <extra server_cases args> -- <server env...>
  local phase="$1" extra="$2"; shift 3
  echo "=== phase: $phase ==="
  start_server "$OUT/$phase.log" "$@"
  if [[ "$phase" != "warmup" ]] && ! wait_ready; then
    echo "the server did not become ready (log: $OUT/$phase.log)"; rc=1; stop_server; return
  fi
  python3 "$HERE/server_cases.py" --port "$PORT" --phase "$phase" $extra || rc=1
  if [[ "$phase" == "cancel-race" ]]; then
    # the worker log must show the second check stopped the task and that the
    # model never ran for it (only the "after race" job ran)
    if grep -q "skipping cancelled task 1 (after claim)" "$OUT/$phase.log" \
       && ! grep -q "MockVideoRunner\] task 1 " "$OUT/$phase.log"; then
      echo "PASS  race: worker log: skipped after claim, the runner was not started"
    else
      echo "FAIL  race: the worker log does not show a skip after claim"
      grep -h "task 1\|skipping" "$OUT/$phase.log" | head -5
      rc=1
    fi
  fi
  stop_server
}

run_contract() {
  echo "=== phase: contract ==="
  start_server "$OUT/contract.log" TT_VIDEO_MOCK_DELAY_MS=300 TT_VIDEO_MOCK_WRITE_FILE=1
  if ! wait_ready; then
    echo "the server did not become ready (log: $OUT/contract.log)"; rc=1; stop_server; return
  fi
  python3 "$HERE/contract_cases.py" --base-url "$BASE" --require-file || rc=1
  stop_server
}

# a real mp4 made by ffmpeg when it is installed (checks the faststart remux), else a fake one
WRITE=1; command -v ffmpeg >/dev/null 2>&1 && WRITE=ffmpeg
run_contract
run_phase main "--max-queue 3" -- TT_VIDEO_MOCK_DELAY_MS=3000 TT_VIDEO_MOCK_WRITE_FILE=$WRITE MAX_QUEUE_SIZE=3
run_phase sync-error "" -- MAX_QUEUE_SIZE=3
run_phase limits "" -- MAX_JOBS=2 MAX_QUEUE_SIZE=3
run_phase warmup "" -- TT_VIDEO_MOCK_WARMUP_MS=6000
# The race is between a worker process and the service (the shm queue and the
# cancel marker); in process mode there is no such window, so this phase pins
# the worker process.
run_phase cancel-race "" -- TT_VIDEO_IN_PROCESS=0 TT_MEDIA_TEST_PRECLAIM_DELAY_MS=1500 TT_VIDEO_MOCK_DELAY_MS=3000
exit $rc
