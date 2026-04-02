#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# End-to-end perf smoke: mock external video runner (~TT_MOCK_VIDEO_TARGET_SECONDS
# simulated generation, default 10s) + server with sp_runner (SHM). External runner
# builds frames and export_to_mp4 immediately; SHM output carries the .mp4 path only.
# Submits E2E_NUM_JOBS sequential video jobs (default 5), records concise timings per run + means.
#
# A/B testing: run once at baseline commit 9ad2052 (feat(vid-shm): Add tests) and
# once at your final HEAD — same TT_MOCK_VIDEO_TARGET_SECONDS, MODEL, PORT.
# Save metrics: ./performance/snapshot_prometheus_metrics.sh metrics-<label>.txt
#
# Prerequisites:
#   - Run from a machine where TT device workers can start (same as normal server).
#   - Python env on PATH (e.g. tt-metal python_env).
#   - cd: tt-media-server (or pass TT_MEDIA_SERVER_ROOT).
#
# Usage:
#   setup_tt_env changes cwd; invoke this script by absolute path (recommended):
#
#     source ~/setup_tt_env.sh && \
#       /path/to/tt-inference-server/tt-media-server/performance/perf_sp_runner_mock_e2e.sh
#
#   Each run writes a **new** report under performance/reports/… unless you pin a path (see below).
#   A symlink performance/perf-sp-runner-mock-e2e-report-LATEST.txt → latest report (same repo tree).
#
# Env (optional):
#   TT_MOCK_OUTPUT_NUM_FRAMES      default 81 (Wan22 quad max; override for longer tensors)
#   TT_MOCK_VIDEO_TARGET_SECONDS   default 10 (total mock sleep; independent of output size)
#   TT_MOCK_VIDEO_RAMP             up (default) | flat — see mock_video_runner.py
#   TT_MOCK_FLOAT32_TENSOR         default 1 — stacked tensor float32 [0,1) (~0.83 GiB @ 81×720×1280×3;
#                                  capped by MAX_VIDEO_SIZE in config/constants.py). Set 0 for uint8 mock.
#   TT_MOCK_SIMULATE_30S_1080P     set to 1 for 1920×1080 × 480 frames (~30s@16fps tensor;
#                                  ~3GiB — heavy). Omit for default API/request dimensions.
#   TT_MOCK_OUTPUT_{HEIGHT,WIDTH,NUM_FRAMES}  optional per-axis overrides (see mock_video_runner.py).
#                                  Defaults: Wan22 quad max res 81×720×1280 (see TTWan22Runner in dit_runners).
#   SERVICE_PORT                   default 8000
#   API_KEY                        default your-secret-key
#   MODEL                          omitted by default: Wan name would map to tt-wan2.2 in
#                                  settings._set_config_overrides before sp_runner. Set
#                                  E2E_EXPORT_MODEL=1 to export Wan2.2-T2V-A14B-Diffusers.
#   HF_TOKEN MODEL JWT_SECRET      set empty for local if needed
#   ENABLE_TELEMETRY               default true
#   METRICS_SNAPSHOT               optional path for Prometheus scrape (see snapshot_prometheus_metrics.sh)
#   PERF_E2E_REPORT                optional hint path (never overwrites that file):
#                                    unset → TT_MEDIA_SERVER_ROOT/performance/reports/perf-sp-runner-mock-e2e-<stamp>.txt
#                                    dir   → <dir>/perf-sp-runner-mock-e2e-<stamp>.txt
#                                    file  → <dir>/<basename-without-.txt>-<stamp>.txt next to that file
#   PERF_E2E_REPORT_EXACT          if set, write exactly this file path (overwrites). For CI compare only.
#   PERF_E2E_SYMLINK_LATEST        default 1: update performance/perf-sp-runner-mock-e2e-report-LATEST.txt
#                                  in TT_MEDIA_SERVER_ROOT when the report lives under that tree.
#   E2E_NUM_JOBS                   default 5 — sequential POST / poll jobs for means in report
#   Per-job TSV (${TMPDIR}/perf_e2e_runs_*.tsv): includes client_ttf_prog_s, mock_ttft_first_frame_s
#   (encoder TTFT from [VIDEO_DELIVERY]), mock_encode_after_first_s; report appends serving-style P99 block.

set -euo pipefail

TTMS="${TT_MEDIA_SERVER_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$TTMS" || exit 1

# Unique human-readable report path each run (no accidental overwrite of perf-sp-runner-mock-e2e-report.txt).
e2e_resolve_perf_report_path() {
  local stamp dir base parent r
  stamp="$(date +%Y%m%d-%H%M%S)-$$"
  if [[ -n "${PERF_E2E_REPORT_EXACT:-}" ]]; then
    r="${PERF_E2E_REPORT_EXACT}"
    [[ "$r" = /* ]] || r="${TTMS}/${r}"
    echo "$r"
    return
  fi
  if [[ -n "${PERF_E2E_REPORT:-}" ]]; then
    r="${PERF_E2E_REPORT}"
    [[ "$r" = /* ]] || r="${TTMS}/${r}"
    if [[ -d "$r" ]]; then
      mkdir -p "$r"
      echo "${r%/}/perf-sp-runner-mock-e2e-${stamp}.txt"
      return
    fi
    parent=$(dirname "$r")
    base=$(basename "$r" .txt)
    mkdir -p "$parent"
    echo "${parent}/${base}-${stamp}.txt"
    return
  fi
  mkdir -p "${TTMS}/performance/reports"
  echo "${TTMS}/performance/reports/perf-sp-runner-mock-e2e-${stamp}.txt"
}

export PERF_E2E_REPORT
PERF_E2E_REPORT="$(e2e_resolve_perf_report_path)"
echo "==> E2E report path: ${PERF_E2E_REPORT}"

PYTHON="${PYTHON:-python3}"
PORT="${SERVICE_PORT:-8000}"
API_KEY="${API_KEY:-your-secret-key}"
API_BASE="http://127.0.0.1:${PORT}"
MOCK_TARGET="${TT_MOCK_VIDEO_TARGET_SECONDS:-10}"
SHM_IN="${TT_VIDEO_SHM_INPUT:-tt_video_in_perf_e2e}"
SHM_OUT="${TT_VIDEO_SHM_OUTPUT:-tt_video_out_perf_e2e}"

export ENABLE_TELEMETRY="${ENABLE_TELEMETRY:-true}"
export HF_TOKEN="${HF_TOKEN:-}"
export JWT_SECRET="${JWT_SECRET:-}"
export MODEL_RUNNER=sp_runner
# Wan in MODEL matches both tt-wan2.2 and sp_runner; map iteration picks tt-wan2.2 first.
unset MODEL
if [[ "${E2E_EXPORT_MODEL:-}" == "1" ]]; then
  export MODEL="${MODEL:-Wan2.2-T2V-A14B-Diffusers}"
fi
export TT_VIDEO_SHM_INPUT="$SHM_IN"
export TT_VIDEO_SHM_OUTPUT="$SHM_OUT"
export TT_MOCK_VIDEO_TARGET_SECONDS="$MOCK_TARGET"
# Match ~1 GiB float32 budget (MAX_VIDEO_SIZE): 1280×720×3×81×4 < 1024**3.
export TT_MOCK_FLOAT32_TENSOR="${TT_MOCK_FLOAT32_TENSOR:-1}"
# Wan22 quad-mesh max output (TTWan22Runner when mesh shape is 4×8): 81×720×1280.
export TT_MOCK_OUTPUT_NUM_FRAMES="${TT_MOCK_OUTPUT_NUM_FRAMES:-81}"
export TT_MOCK_OUTPUT_HEIGHT="${TT_MOCK_OUTPUT_HEIGHT:-720}"
export TT_MOCK_OUTPUT_WIDTH="${TT_MOCK_OUTPUT_WIDTH:-1280}"

# Prometheus multiprocess (worker + main) — must exist before prometheus_client import
export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus_multiproc_e2e}"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

UVICORN_LOG="${TMPDIR:-/tmp}/perf_sp_runner_mock_e2e_uvicorn.log"
MOCK_LOG="${TMPDIR:-/tmp}/mock_video_runner_e2e.log"
E2E_NUM_JOBS="${E2E_NUM_JOBS:-5}"
E2E_RUNS_TSV="${TMPDIR:-/tmp}/perf_e2e_runs_$$.tsv"
rm -f "$UVICORN_LOG"
export E2E_RUNS_TSV E2E_NUM_JOBS

cleanup() {
  if [[ -n "${UVICORN_PID:-}" ]] && kill -0 "$UVICORN_PID" 2>/dev/null; then
    kill "$UVICORN_PID" 2>/dev/null || true
    wait "$UVICORN_PID" 2>/dev/null || true
  fi
  if [[ -n "${MOCK_PID:-}" ]] && kill -0 "$MOCK_PID" 2>/dev/null; then
    kill "$MOCK_PID" 2>/dev/null || true
    wait "$MOCK_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "==> Starting mock video runner (SHM create=True, ~${MOCK_TARGET}s mock sleep; output ${TT_MOCK_OUTPUT_NUM_FRAMES}×${TT_MOCK_OUTPUT_HEIGHT}×${TT_MOCK_OUTPUT_WIDTH}; TT_MOCK_FLOAT32_TENSOR=${TT_MOCK_FLOAT32_TENSOR:-})..."
"$PYTHON" -m tt_model_runners.mock_video_runner >>"$MOCK_LOG" 2>&1 &
MOCK_PID=$!
sleep 2

echo "==> Starting uvicorn (sp_runner + video service) on port ${PORT}..."
echo "    Uvicorn log: ${UVICORN_LOG}"
"$PYTHON" -m uvicorn main:app --host 127.0.0.1 --port "$PORT" --lifespan on >>"$UVICORN_LOG" 2>&1 &
UVICORN_PID=$!

echo "==> Waiting for server (port ${PORT})..."
for _ in $(seq 1 90); do
  if curl -sS -o /dev/null "http://127.0.0.1:${PORT}/metrics" 2>/dev/null; then
    break
  fi
  sleep 1
done

if ! curl -sS -o /dev/null "http://127.0.0.1:${PORT}/metrics" 2>/dev/null; then
  echo "Server did not become ready; see logs. Mock log: ${MOCK_LOG}"
  exit 1
fi

extractTimingField() {
  local line="$1" key="$2"
  echo "$line" | grep -o "${key}=[0-9.]*" 2>/dev/null | head -1 | cut -d= -f2
}

appendE2eRunRow() {
  local runId="$1" wallSec="$2" postSec="$3" ttfProgSec="$4" pollN="$5"
  local spLine procSec extWall phaseA mockExport mockFrames encM after benchHandoff
  local vdLine mockTtftFirst mockEncAfterFirst
  spLine=$(grep '\[SP\] Timing:' "$UVICORN_LOG" 2>/dev/null | sed -n "${runId}p" || true)
  procSec=$(grep '\[process_request\] async executed in' "$UVICORN_LOG" 2>/dev/null | sed -n "${runId}p" | sed -n 's/.*executed in \([0-9.]*\) seconds.*/\1/p')
  extWall=$(extractTimingField "$spLine" external_runner_wall_s)
  phaseA=$(extractTimingField "$spLine" job_process_start_to_shm_input_write_wall_s)
  mockExport=$(grep '\[export_to_mp4\] executed in' "$MOCK_LOG" 2>/dev/null | tail -1 | sed -n 's/.*executed in \([0-9.]*\) seconds.*/\1/p')
  mockFrames=$(grep '\[_process_frames_for_export\] executed in' "$MOCK_LOG" 2>/dev/null | tail -1 | sed -n 's/.*executed in \([0-9.]*\) seconds.*/\1/p')
  benchHandoff=$(grep '\[MOCK_AFTER_BENCH\]' "$MOCK_LOG" 2>/dev/null | sed -n "${runId}p" | sed -n 's/.*bench_to_handoff_s=\([0-9.]*\).*/\1/p')
  vdLine=$(grep '\[VIDEO_DELIVERY\] task_id=' "$MOCK_LOG" 2>/dev/null | sed -n "${runId}p" || true)
  mockTtftFirst=$(echo "$vdLine" | sed -n 's/.*ttft_to_first_frame_appended_s=\([0-9.]*\).*/\1/p')
  mockEncAfterFirst=$(echo "$vdLine" | sed -n 's/.*encode_after_first_frame_s=\([0-9.]*\).*/\1/p')
  [[ -n "$procSec" ]] || procSec="0"
  [[ -n "$extWall" ]] || extWall="0"
  [[ -n "$phaseA" ]] || phaseA="0"
  [[ -n "$mockExport" ]] || mockExport="0"
  [[ -n "$mockFrames" ]] || mockFrames="0"
  [[ -n "$benchHandoff" ]] || benchHandoff="0"
  [[ -n "$mockTtftFirst" ]] || mockTtftFirst="n/a"
  [[ -n "$mockEncAfterFirst" ]] || mockEncAfterFirst="n/a"
  encM=$(awk -v a="$mockExport" -v b="$mockFrames" 'BEGIN { printf "%.4f", a - b }')
  after=$(awk -v p="$procSec" -v e="$extWall" 'BEGIN { printf "%.4f", p - e }')
  printf '%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$runId" "$wallSec" "$postSec" "$ttfProgSec" "$mockTtftFirst" "$mockEncAfterFirst" "$pollN" "$procSec" "$extWall" "$phaseA" "$mockExport" "$encM" "$after" "$benchHandoff" >>"$E2E_RUNS_TSV"
}

echo "run,wall_s,post_s,client_ttf_prog_s,mock_ttft_first_frame_s,mock_encode_after_first_s,polls,proc_s,ext_s,phase_a_shm_s,mock_exp_s,enc_mock_s,after_s,bench_handoff_s" >"$E2E_RUNS_TSV"

echo "==> Submitting ${E2E_NUM_JOBS} video job(s) (each: mock ~${MOCK_TARGET}s + encode + post; curl poll ~2s)..."
TOTAL_WALL_SUM=0
for RUN_IDX in $(seq 1 "$E2E_NUM_JOBS"); do
  T0=$(date +%s)
  REQ_TMP=$(mktemp)
  E2E_T0_FLOAT=$("$PYTHON" -c 'import time; print(time.time())')
  export E2E_T0_FLOAT
  POST_S=$(curl -sS -X POST "${API_BASE}/v1/videos/generations" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"perf e2e smoke","num_inference_steps":12,"height":720,"width":1280,"num_frames":81}' \
    -o "$REQ_TMP" -w '%{time_total}')
  JOB_ID=$("$PYTHON" -c "import json; print(json.load(open('$REQ_TMP'))['id'])")
  rm -f "$REQ_TMP"
  echo "    run ${RUN_IDX}/${E2E_NUM_JOBS} job id: ${JOB_ID}"

  DEADLINE=$((T0 + MOCK_TARGET + 180))
  STATUS=""
  TTF_PROG="n/a"
  POLL_N=0
  while true; do
    NOW=$(date +%s)
    if (( NOW > DEADLINE )); then
      echo "Timeout waiting for job (run ${RUN_IDX})"
      curl -sS "${API_BASE}/v1/videos/generations/${JOB_ID}" -H "Authorization: Bearer ${API_KEY}" || true
      exit 1
    fi
    META=$(curl -sS "${API_BASE}/v1/videos/generations/${JOB_ID}" -H "Authorization: Bearer ${API_KEY}")
    STATUS=$(echo "$META" | "$PYTHON" -c 'import sys,json; print(json.load(sys.stdin).get("status",""))')
    POLL_N=$((POLL_N + 1))
    if [[ "$STATUS" == "in_progress" && "$TTF_PROG" == "n/a" ]]; then
      TTF_PROG=$("$PYTHON" -c 'import os, time; print("%.4f" % (time.time() - float(os.environ["E2E_T0_FLOAT"])))')
    fi
    if [[ "$STATUS" == "completed" ]]; then
      break
    fi
    if [[ "$STATUS" == "failed" ]] || [[ "$STATUS" == "cancelled" ]]; then
      echo "$META"
      exit 1
    fi
    sleep 2
  done
  unset E2E_T0_FLOAT
  T1=$(date +%s)
  WALL=$((T1 - T0))
  TOTAL_WALL_SUM=$((TOTAL_WALL_SUM + WALL))
  echo "    run ${RUN_IDX} wall ${WALL}s (status=${STATUS}) post=${POST_S}s ttf_prog=${TTF_PROG}s polls=${POLL_N}"
  appendE2eRunRow "$RUN_IDX" "$WALL" "$POST_S" "$TTF_PROG" "$POLL_N"
done

WALL=$(awk -v s="$TOTAL_WALL_SUM" -v n="$E2E_NUM_JOBS" 'BEGIN { printf "%.1f", s / n }')
echo "==> All ${E2E_NUM_JOBS} job(s) done (mean client wall ~${WALL}s)"

echo "==> Sample metrics (model_inference + post_processing histograms)..."
curl -sS "http://127.0.0.1:${PORT}/metrics" | grep -E 'tt_media_server_(model_inference|post_processing)_duration_seconds' | grep 'sp_runner' | head -40 || true

if [[ -n "${METRICS_SNAPSHOT:-}" ]]; then
  echo "==> Writing Prometheus snapshot to ${METRICS_SNAPSHOT}..."
  SNAP="${TTMS}/performance/snapshot_prometheus_metrics.sh"
  if [[ ! -x "$SNAP" ]]; then
    chmod +x "$SNAP" 2>/dev/null || true
  fi
  SERVICE_PORT="$PORT" "$SNAP" "$METRICS_SNAPSHOT"
fi

writePerfE2eReport() {
  local reportFile pickleLine sizeRaw sizeBytes mib processSec exportSec shapeTuple metricsRel metricsLines
  local spTimingLine extWall pickleLoad cleanupS framesProcSec postPipeline spRunnerLocal remainder encodeApprox
  local jobToShm queueToShm buildReq preQueueFromJob mockExportSec mockFramesProcSec encodeApproxMock
  local perRunMd meansMd tsvPath videoDeliveryLine mockNonSleepLine mockAfterBenchLine
  local lastBenchHandoff benchPlusServerAfter benchHandoffMean benchPlusServerMean afterMean
  local servingSummaryBlock
  reportFile="${PERF_E2E_REPORT:?PERF_E2E_REPORT must be set}"
  [[ "$reportFile" = /* ]] || reportFile="${TTMS}/${reportFile}"
  tsvPath="${E2E_RUNS_TSV:-}"
  perRunMd=""
  meansMd="| **mean** | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
  servingSummaryBlock=""
  if [[ -n "$tsvPath" && -f "$tsvPath" ]]; then
    perRunMd=$(awk -F, 'NR==1{next} { printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14 }' "$tsvPath")
    meansMd=$(awk -F, '
NR==1 { next }
{
  w+=$2+0; pst+=$3+0; polls+=$7+0; p+=$8+0; ex+=$9+0; pa+=$10+0; mx+=$11+0; em+=$12+0; af+=$13+0; bh+=$14+0; n++
  if ($4 != "n/a") { ttf+=$4+0; nz++ }
  if ($5 != "n/a") { mttf+=$5+0; nz5++ }
  if ($6 != "n/a") { mea+=$6+0; nz6++ }
}
END {
  if (n < 1) { print "| **mean** | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"; exit }
  ttfstr = (nz > 0) ? sprintf("%.4f", ttf / nz) : "n/a"
  mttfstr = (nz5 > 0) ? sprintf("%.4f", mttf / nz5) : "n/a"
  meatr = (nz6 > 0) ? sprintf("%.4f", mea / nz6) : "n/a"
  printf "| **mean (n=%d)** | %.2f | %.4f | %s | %s | %s | %.2f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |\n", n, w/n, pst/n, ttfstr, mttfstr, meatr, polls/n, p/n, ex/n, pa/n, mx/n, em/n, af/n, bh/n
}' "$tsvPath")
    benchHandoffMean=$(awk -F, 'NR>1 {s+=$14+0; n++} END { if (n > 0) printf "%.4f", s / n; else print "n/a" }' "$tsvPath")
    afterMean=$(awk -F, 'NR>1 {s+=$13+0; n++} END { if (n > 0) printf "%.4f", s / n; else print "n/a" }' "$tsvPath")
    benchPlusServerMean=$(awk -F, 'NR>1 {s+=$14+$13+0; n++} END { if (n > 0) printf "%.4f", s / n; else print "n/a" }' "$tsvPath")
    export E2E_RUNS_TSV="$tsvPath"
    export E2E_NUM_JOBS
    export TT_MOCK_OUTPUT_NUM_FRAMES
    servingSummaryBlock=$("$PYTHON" "$TTMS/performance/perf_e2e_serving_summary.py")
  else
    benchHandoffMean="n/a"
    afterMean="n/a"
    benchPlusServerMean="n/a"
  fi

  pickleLine=$(grep -E 'Received (file_path from SHM output|mp4 path from SHM):' "$UVICORN_LOG" 2>/dev/null | tail -1 || true)
  sizeRaw=$(echo "$pickleLine" | sed -n 's/.*size=\([0-9,]*\) bytes.*/\1/p')
  if [[ -n "$sizeRaw" ]]; then
    sizeBytes=$(echo "$sizeRaw" | tr -d ',')
    mib=$(awk -v b="$sizeBytes" 'BEGIN { printf "%.1f", b / 1024 / 1024 }')
    sizeRaw="${sizeRaw} bytes (~${mib} MiB)"
  else
    sizeRaw="n/a"
  fi

  processSec=$(grep '\[process_request\] async executed in' "$UVICORN_LOG" 2>/dev/null | tail -1 | sed -n 's/.*executed in \([0-9.]*\) seconds.*/\1/p')
  [[ -n "$processSec" ]] || processSec="n/a"

  # Uvicorn: last export line is often VideoPostprocessing *warmup* (tiny tensor), not the job, when the
  # device runner already returned an .mp4 path (str branch in video_service.video_worker_function).
  exportSec=$(grep '\[export_to_mp4\] executed in' "$UVICORN_LOG" 2>/dev/null | tail -1 | sed -n 's/.*executed in \([0-9.]*\) seconds.*/\1/p')
  [[ -n "$exportSec" ]] || exportSec="n/a"

  framesProcSec=$(grep '\[_process_frames_for_export\] executed in' "$UVICORN_LOG" 2>/dev/null | tail -1 | sed -n 's/.*executed in \([0-9.]*\) seconds.*/\1/p')
  [[ -n "$framesProcSec" ]] || framesProcSec="n/a"

  mockExportSec=$(grep '\[export_to_mp4\] executed in' "$MOCK_LOG" 2>/dev/null | tail -1 | sed -n 's/.*executed in \([0-9.]*\) seconds.*/\1/p')
  [[ -n "$mockExportSec" ]] || mockExportSec="n/a"
  mockFramesProcSec=$(grep '\[_process_frames_for_export\] executed in' "$MOCK_LOG" 2>/dev/null | tail -1 | sed -n 's/.*executed in \([0-9.]*\) seconds.*/\1/p')
  [[ -n "$mockFramesProcSec" ]] || mockFramesProcSec="n/a"

  spTimingLine=$(grep '\[SP\] Timing:' "$UVICORN_LOG" 2>/dev/null | tail -1 || true)
  jobToShm=$(extractTimingField "$spTimingLine" job_process_start_to_shm_input_write_wall_s)
  queueToShm=$(extractTimingField "$spTimingLine" device_queue_to_shm_input_write_wall_s)
  buildReq=$(extractTimingField "$spTimingLine" sp_runner_build_request_s)
  extWall=$(extractTimingField "$spTimingLine" external_runner_wall_s)
  pickleLoad=$(extractTimingField "$spTimingLine" pickle_load_s)
  cleanupS=$(extractTimingField "$spTimingLine" sp_runner_cleanup_s)
  [[ -n "$jobToShm" ]] || jobToShm="n/a"
  [[ -n "$queueToShm" ]] || queueToShm="n/a"
  [[ -n "$buildReq" ]] || buildReq="n/a"
  [[ -n "$extWall" ]] || extWall="n/a"
  [[ -n "$pickleLoad" ]] || pickleLoad="n/a"
  [[ -n "$cleanupS" ]] || cleanupS="n/a"

  preQueueFromJob="n/a"
  if [[ "$jobToShm" != n/a && "$queueToShm" != n/a ]]; then
    preQueueFromJob=$(awk -v j="$jobToShm" -v q="$queueToShm" 'BEGIN { printf "%.4f", j - q }')
  fi

  postPipeline="n/a"
  spRunnerLocal="n/a"
  remainder="n/a"
  encodeApprox="n/a"
  if [[ "$processSec" != n/a && "$extWall" != n/a ]]; then
    postPipeline=$(awk -v pr="$processSec" -v ex="$extWall" 'BEGIN { printf "%.4f", pr - ex }')
  fi
  if [[ "$pickleLoad" != n/a && "$cleanupS" != n/a ]]; then
    spRunnerLocal=$(awk -v a="$pickleLoad" -v b="$cleanupS" 'BEGIN { printf "%.4f", a + b }')
  fi
  if [[ "$processSec" != n/a && "$extWall" != n/a && "$pickleLoad" != n/a && "$cleanupS" != n/a && "$exportSec" != n/a ]]; then
    remainder=$(awk -v pr="$processSec" -v ex="$extWall" -v pl="$pickleLoad" -v cl="$cleanupS" -v ep="$exportSec" \
      'BEGIN { printf "%.4f", pr - ex - pl - cl - ep }')
  fi
  if [[ "$exportSec" != n/a && "$framesProcSec" != n/a ]]; then
    encodeApprox=$(awk -v ep="$exportSec" -v fp="$framesProcSec" 'BEGIN { printf "%.4f", ep - fp }')
  fi
  encodeApproxMock="n/a"
  if [[ "$mockExportSec" != n/a && "$mockFramesProcSec" != n/a ]]; then
    encodeApproxMock=$(awk -v ep="$mockExportSec" -v fp="$mockFramesProcSec" 'BEGIN { printf "%.4f", ep - fp }')
  fi

  shapeTuple=$(grep '\[SP\] LOADED video' "$UVICORN_LOG" 2>/dev/null | tail -1 | grep -o 'shape=([^)]*)' | sed 's/^shape=//' || true)
  if [[ -z "$shapeTuple" ]]; then
    shapeTuple=$(grep '\[MOCK_BENCH\].*full tensor in RAM' "$MOCK_LOG" 2>/dev/null | tail -1 | grep -oE 'shape=\([^)]+\)' | sed 's/^shape=//' || true)
  fi
  videoDeliveryLine=$(grep '\[VIDEO_DELIVERY\] task_id=' "$MOCK_LOG" 2>/dev/null | tail -1 || true)
  mockNonSleepLine=$(grep '\[MOCK\] non_sleep_pipeline:' "$MOCK_LOG" 2>/dev/null | tail -1 || true)
  mockAfterBenchLine=$(grep '\[MOCK_AFTER_BENCH\]' "$MOCK_LOG" 2>/dev/null | tail -1 || true)
  lastBenchHandoff=$(echo "$mockAfterBenchLine" | sed -n 's/.*bench_to_handoff_s=\([0-9.]*\).*/\1/p')
  benchPlusServerAfter="n/a"
  if [[ -n "$lastBenchHandoff" && "$postPipeline" != n/a ]]; then
    benchPlusServerAfter=$(awk -v b="$lastBenchHandoff" -v a="$postPipeline" 'BEGIN { printf "%.4f", b + a }')
  fi
  if [[ -z "$shapeTuple" ]]; then
    shapeTuple="(1, ${TT_MOCK_OUTPUT_NUM_FRAMES}, ${TT_MOCK_OUTPUT_HEIGHT:-720}, ${TT_MOCK_OUTPUT_WIDTH:-1280}, 3) (from env; preset overrides not reflected)"
  fi

  metricsRel="${METRICS_SNAPSHOT:-}"
  if [[ -n "$metricsRel" && -f "${TTMS}/${metricsRel}" ]]; then
    metricsLines=$(wc -l <"${TTMS}/${metricsRel}" | tr -d ' ')
  elif [[ -n "$metricsRel" && -f "$metricsRel" ]]; then
    metricsLines=$(wc -l <"$metricsRel" | tr -d ' ')
  else
    metricsLines="n/a"
    [[ -n "$metricsRel" ]] || metricsRel="(METRICS_SNAPSHOT not set)"
  fi

  local nfDisplay hDisplay wDisplay approxGapMock
  nfDisplay="${TT_MOCK_OUTPUT_NUM_FRAMES:-81}"
  hDisplay="${TT_MOCK_OUTPUT_HEIGHT:-720}"
  wDisplay="${TT_MOCK_OUTPUT_WIDTH:-1280}"
  set +e
  if [[ "$shapeTuple" =~ \(1,\ *([0-9]+),\ *([0-9]+),\ *([0-9]+),\ *3\) ]]; then
    nfDisplay="${BASH_REMATCH[1]}"
    hDisplay="${BASH_REMATCH[2]}"
    wDisplay="${BASH_REMATCH[3]}"
  fi
  set -e
  approxGapMock=$(awk -v w="${WALL:-0}" -v m="${MOCK_TARGET:-10}" 'BEGIN { d = w - m; if (d < 0) d = 0; printf "%.0f", d }')

  mkdir -p "$(dirname "$reportFile")"
  cat >"$reportFile" <<EOF
# perf_sp_runner_mock_e2e — full report

Report file: \`${reportFile}\`

Run finished successfully (${E2E_NUM_JOBS:-1} sequential jobs).

## What this measures (pipeline)

End-to-end path for one video job with \`MODEL_RUNNER=sp_runner\` and **MP4-first / immediate export** in the external runner:

1. SHM **input**: \`VideoRequest\` metadata only (prompt, dims, steps, …).
2. **External runner** (e.g. \`mock_video_runner\`): generates frames, then **\`VideoManager.export_to_mp4\` inside that process** — no large tensor is sent back on SHM.
3. SHM **output**: \`VideoResponse\` with **filesystem path** to the finished \`.mp4\` (and optional size in logs).
4. **Device worker (\`SPRunner\`)**: \`read_response\` returns the path; \`pickle_load_s\` / \`sp_runner_cleanup_s\` are **0** (no server-side pickle of frames).
5. **VideoPostprocessing** (\`video_service.video_worker_function\`): receives a **\`str\` path** → **passthrough** to final job path (no second full encode on the server for that job).

**Phase A — Server: job → SHM input** (wall \`time.time()\`):

- \`job_process_start_to_shm_input_write_wall_s\`: \`JobManager\` start of handling → device worker \`write_request\`.
- \`device_queue_to_shm_input_write_wall_s\`: \`task_queue.put\` → \`write_request\`.
- \`sp_runner_build_request_s\`: worker-only \`VideoRequest\` build (\`perf_counter\`).

**Phase B — External runner (all heavy work here):** \`external_runner_wall_s\` — time blocked in \`read_response\` while the external process runs **mock delay + frame gen + \`export_to_mp4\`** (encode is **not** split out in \`[SP] Timing:\`; see **mock log** for \`export_to_mp4\` / \`_process_frames_for_export\`).

**Phase C — Server: after SHM output path is ready:** return path through \`process_request\`; post worker validates/hands off path only (uvicorn \`export_to_mp4\` lines are often **warmup**, not this job — ignore for encode comparison).

**Headline slice after the external runner returns:**

\`after_shm_output_path_ready_s = process_request_s − external_runner_wall_s\`

(server-side tail from “MP4 path known” → end of \`process_request\`).

\`\`\`mermaid
sequenceDiagram
  participant Job as job / process_request
  participant Sch as scheduler task_queue
  participant SP as SPRunner.run
  participant Ext as external runner
  participant Post as VideoPostprocessing

  Job->>Sch: enqueue request
  Sch->>SP: worker dequeues
  Note over Job,SP: Phase A wall stamps
  SP->>Ext: SHM write_request VideoRequest
  Note over Ext: frames + export_to_mp4 immediately
  Ext-->>SP: SHM response: .mp4 path
  Note over SP: external_runner_wall_s
  SP-->>Job: path str
  Job->>Post: post_process path passthrough
  Post-->>Job: final path
\`\`\`

## Client / script wall clock

| Item | Value |
|------|-------|
| Jobs submitted | ${E2E_NUM_JOBS:-1} |
| Mean wall time per job (script \`curl\` poll, ~2 s granularity) | ${WALL} s |
| Mock sleep budget per job (env \`TT_MOCK_VIDEO_TARGET_SECONDS\`) | ${MOCK_TARGET} s |
| Approx. gap vs mock (mean wall − budget) | ~${approxGapMock} s |

## After \`[MOCK_BENCH] full tensor in RAM\` → server → client (highlight)

Clock starts **immediately after** the mock runner logs that the full frame tensor exists (end of synthetic / bench phase). This section chains **mock handoff**, **server tail inside \`process_request\`**, and what is **not** measured to the browser.

| Stage | Last job (s) | Mean (all runs) | Meaning |
|------|-------------:|----------------:|---------|
| **bench→handoff** (mock) | ${lastBenchHandoff:-n/a} | ${benchHandoffMean} | Encode \`.mp4\` + SHM \`write_response\` (see \`[MOCK_AFTER_BENCH]\` line below). |
| **+ server after path** | ${postPipeline} | ${afterMean} | \`process_request − external_runner_wall\` — handler after \`read_response\` returns \`.mp4\` path. |
| **= mock+server subtotal** | ${benchPlusServerAfter} | ${benchPlusServerMean} | \`bench_handoff_s + after_s\` per run, then mean — lower bound for “tensor ready in mock” → **server finished this request**. |
| **→ end user \`completed\`** | not in table | — | Client script polls every ~2 s; **add up to ~2 s** after server completes before \`GET\` returns \`completed\`. Not correlated to mock monotonic clock. |

Raw **last job** \`[MOCK_AFTER_BENCH]\`:

\`\`\`
${mockAfterBenchLine:-n/a}
\`\`\`

## Per-run timings (concise)

| # | wall | post | ttf_prog | enc_ttft | enc_after_1st | polls | proc | ext | pA_shm | mock exp | enc~mock | after | bench→handoff |
|---|-----:|-----:|---------:|---------:|---------------:|------:|-----:|----:|-------:|---------:|---------:|------:|---------------:|
${perRunMd}
${meansMd}

Raw per-job TTFT (seconds): **ttf_prog** = client API; **enc_ttft** = \`ttft_to_first_frame_appended_s\` from \`[VIDEO_DELIVERY]\`; **enc_after_1st** = \`encode_after_first_frame_s\` (full TSV path: \`${E2E_RUNS_TSV:-}\`).

- **bench→handoff** — \`bench_to_handoff_s\` from mock log \`[MOCK_AFTER_BENCH]\`: wall time **starting right after** the \`[MOCK_BENCH] … full tensor in RAM\` line until the **SHM success response is written** (encode + write \`.mp4\` + \`write_response\`). This is the external-runner slice *after* synthetic generation; it does **not** include server tail or client polling.

- **wall** — client integer seconds from **start of POST** to \`completed\` (poll step ~2 s; coarser than sub-second columns).
- **post** — \`curl\` time for create job POST (\`time_total\`).
- **ttf_prog** — **client “TTFT-like”**: seconds from **same clock as POST start** to **first GET** that sees \`status=in_progress\` (\`n/a\` if the job never appeared as \`in_progress\` before \`completed\`, or timing bug). Column \`client_ttf_prog_s\` in TSV.
- **enc_ttft** — copies \`ttft_to_first_frame_appended_s\` (\`mock_ttft_first_frame_s\`). **Batch FFmpeg:** full \`export_to_mp4\` wall (legacy name; not first-frame TTFT). **Incremental imageio:** time to first \`append_data\`.
- **enc_after_1st** — \`encode_after_first_frame_s\` (\`mock_encode_after_first_s\`). **Batch FFmpeg:** entire ffmpeg time. TPOT-like = value/(frames-1) is synthetic for batch.
- **polls** — number of status GETs before \`completed\`.
- **proc** — \`[process_request]\` total (s), **N-th** line matching this run index (sequential jobs).
- **ext** — \`external_runner_wall_s\` from **that run’s** \`[SP] Timing:\` line (mock sleep + gen + mock \`export_to_mp4\`).
- **pA_shm** — \`job_process_start_to_shm_input_write_wall_s\` for that run (Phase A wall slice).
- **mock exp** — mock log \`[export_to_mp4]\` (s), last line after each run.
- **enc~mock** — mock \`export_to_mp4 − _process_frames_for_export\`.
- **after** — \`proc − ext\` only: **server-side** time inside \`process_request\` **after** \`read_response\` returns the \`.mp4\` path (handoff, post worker queue for path passthrough, job completion bookkeeping). It does **not** include client polling, network outside that handler, or work logged only on the mock process after SHM reply (there is none for this path). Sub-10 ms is normal here; **wall − proc** can still be large because of **poll granularity**.

## Run configuration

| Item | Value |
|------|-------|
| Exit code | 0 |
| Port | ${PORT} |
| Frames × H × W | ${nfDisplay} × ${hDisplay} × ${wDisplay} |
| Tensor shape | ${shapeTuple} |
| SHM output artifact (from log: path size) | ${sizeRaw} |
| Metrics snapshot | ${METRICS_SNAPSHOT:-n/a} (${metricsLines} lines) |
| Uvicorn log | ${UVICORN_LOG} |
| Mock runner log | ${MOCK_LOG} |
| Symlink to latest (if enabled) | \`${TTMS}/performance/perf-sp-runner-mock-e2e-report-LATEST.txt\` |

## Mock log — **last job** \`[VIDEO_DELIVERY]\` (encode / TTFT / throughput)

\`\`\`
${videoDeliveryLine:-n/a}
\`\`\`

## Mock log — **last job** \`non_sleep_pipeline\` (synthetic bench only)

\`\`\`
${mockNonSleepLine:-n/a}
\`\`\`

## Log timings (stages for MP4-immediate export) — **last job only** (tail of logs)

**Server (\`[SP] Timing:\` + uvicorn):** queue → SHM input → block until external runner finishes (includes their encode). **Full-frame encode duration:** use **mock log** \`export_to_mp4\` / \`_process_frames_for_export\` (last lines = this run if log is append-only). For **means**, use the **Per-run** table above.

| Stage | Seconds | Where / meaning |
|------|--------:|-----------------|
| **Phase A — to SHM input** | | |
| \`job_process_start_to_shm_input_write_wall_s\` | ${jobToShm} | Wall: job start → \`write_request\` |
| \`device_queue_to_shm_input_write_wall_s\` | ${queueToShm} | Wall: \`task_queue.put\` → \`write_request\` |
| \`pre_device_queue_from_job_start_s\` (derived) | ${preQueueFromJob} | \`job_process_start_to_shm_input − device_queue_to_shm\` |
| \`sp_runner_build_request_s\` | ${buildReq} | Worker \`perf_counter\`: build \`VideoRequest\` |
| **Phase B — external runner (incl. immediate MP4)** | | |
| \`external_runner_wall_s\` | ${extWall} | \`read_response\` block: **mock + generate + \`export_to_mp4\` in external process** |
| **Phase C — server after .mp4 path (no frame pickle)** | | |
| \`pickle_load_s\` | ${pickleLoad} | Expect **0** — no ndarray pickle on MP4 path |
| \`sp_runner_cleanup_s\` | ${cleanupS} | Expect **0** — no pickle unlink for frames |
| \`process_request\` (total) | ${processSec} | Full handler |
| \`_process_frames_for_export\` (uvicorn) | ${framesProcSec} | Often **warmup** only — do not use for job encode |
| \`export_to_mp4\` (uvicorn) | ${exportSec} | Often **warmup** only — server skips job re-encode when result is \`str\` path |
| **Encode breakdown (external / mock log)** | | |
| \`_process_frames_for_export\` (mock) | ${mockFramesProcSec} | NumPy prep before encode |
| \`export_to_mp4\` (mock) | ${mockExportSec} | **Job’s** MP4 (ffmpeg pipe or legacy path) |

## Derived

| Derived | Seconds | Formula / meaning |
|---------|--------:|-------------------|
| **after_shm_output_path_ready_s** | ${postPipeline} | \`process_request − external_runner_wall\` — server tail after external runner returned **.mp4 path** |
| **sp_runner_after_output_path_s** | ${spRunnerLocal} | \`pickle_load_s + sp_runner_cleanup_s\` — ~0 on MP4 path |
| **approx. encode+write (uvicorn tail)** | ${encodeApprox} | From uvicorn last \`export_to_mp4\` − \`_process_frames_for_export\` — usually **not** the job |
| **approx. encode+write (mock, this job)** | ${encodeApproxMock} | Mock \`export_to_mp4 − _process_frames_for_export\` — **use for OLD vs NEW encode comparison** |
| **remainder (formula uses uvicorn export)** | ${remainder} | \`process_request − external_wall − pickle_load − cleanup − export_uvicorn\` — low signal when encode was in mock |

${servingSummaryBlock}
## Prometheus (\`tt_media_server_*\` / \`tt_video_pipeline_*\`)

See metrics file above. Typical histograms for this path:

- \`tt_media_server_model_inference_duration_seconds\` — \`model_runner="sp_runner"\`; dominated by \`external_runner_wall_s\` (includes external encode).
- \`tt_media_server_post_processing_duration_seconds\` — post worker; job path passthrough when runner returns \`.mp4\` \`str\`.

Grep the snapshot file, e.g. \`grep post_processing\` / \`grep model_inference\`.

---
Raw \`[SP] Timing:\` line (if present):

\`\`\`
${spTimingLine:-n/a}
\`\`\`
EOF
  echo "==> Wrote report: ${reportFile}"
  if [[ "${PERF_E2E_SYMLINK_LATEST:-1}" != "0" ]] && [[ "$reportFile" == "${TTMS}/"* ]]; then
    ln -sf "${reportFile#${TTMS}/}" "${TTMS}/performance/perf-sp-runner-mock-e2e-report-LATEST.txt"
    echo "==> Symlink updated: ${TTMS}/performance/perf-sp-runner-mock-e2e-report-LATEST.txt"
  fi
  cat "$reportFile"
}

writePerfE2eReport || echo "==> WARN: report generation failed (metrics snapshot and job may still be OK)"

echo "==> Done. Full metrics: curl -s http://127.0.0.1:${PORT}/metrics | less"
echo "    Mock runner log: ${MOCK_LOG}"
