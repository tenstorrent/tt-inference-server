#!/bin/bash
# grid_run.sh <server ip> <B> <outdir> <runtime spec json> [--dry-run]
# Grid A (TTFT, OSL 128/1024) and Grid B (decode, OSL 4096) of the frozen suite for the model in model.env, through run.py --workflow benchmarks --tools vllm,
# concurrency 1 and B. ONE run.py call per grid (the first cell compiles), server never restarted by this script. Run from the tt-inference-server tree
# (in the image: /home/container_app_user/tt-inference-server) with the client venv. Cells come from grid_cells.py (the same list the unit tests check).
# Env: MAXISL=<n> drops cells above n (default: all up to max_ctx - 4352; intermediate screens use 131072), GRIDS="A B" (default both).
# --dry-run prints the cell table and the exact run.py commands and writes nothing.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
set -a; . "${HARNESS_MODEL_ENV:-$HERE/model.env}"; set +a
IP=${1:-}; B=${2:-}; OUT=${3:-}; SPEC=${4:-}; DRY=0; [ "${5:-}" = "--dry-run" ] && DRY=1
[ -n "$IP" ] && [ -n "$B" ] && [ -n "$OUT" ] && [ -n "$SPEC" ] || { sed -n 2,9p "$0"; exit 2; }
python3 "$HERE/harness_model.py" key "$B" > /dev/null || exit 2
MAXISL=${MAXISL:-0}; GRIDS=${GRIDS:-"A B"}
TT=${TT_INFERENCE_SERVER_DIR:-$(cd "$HERE/../.." && pwd)}
for G in $GRIDS; do
  P=$(python3 "$HERE/grid_cells.py" pairs $G --maxisl $MAXISL)
  CMD="python3 run.py --model $HARNESS_MODEL_ID --workflow benchmarks --tools vllm --tt-device $HARNESS_DEVICE --dev-mode --no-auth --skip-system-sw-validation --server-url http://$IP:8000 --service-port 8000 --runtime-model-spec-json $SPEC"
  ENVS="TT_BENCH_ISL_OSL_PAIRS=$P TT_BENCH_CONCURRENCIES=1,$B TT_BENCH_NO_STRUCTURED=1 HF_HUB_OFFLINE=1 MODEL_SPECS_ENV=dev CACHE_ROOT=$OUT/grid$G/out"
  if [ $DRY = 1 ]; then
    echo "[dry-run] grid $G B=$B: $(python3 "$HERE/grid_cells.py" table $G $B --maxisl $MAXISL | wc -l) cells (isl osl conc num_prompts):"
    python3 "$HERE/grid_cells.py" table $G $B --maxisl $MAXISL | sed 's/^/[dry-run]   /'
    echo "[dry-run] cd $TT && env $ENVS $CMD > $OUT/grid$G/run.log; rc -> $OUT/grid$G/rc"
    continue
  fi
  mkdir -p "$OUT/grid$G"
  ( cd "$TT" && export TT_BENCH_ISL_OSL_PAIRS=$P TT_BENCH_CONCURRENCIES=1,$B TT_BENCH_NO_STRUCTURED=1 HF_HUB_OFFLINE=1 MODEL_SPECS_ENV=dev CACHE_ROOT=$OUT/grid$G/out
    $CMD < /dev/null > "$OUT/grid$G/run.log" 2>&1; echo $? > "$OUT/grid$G/rc" )
done
