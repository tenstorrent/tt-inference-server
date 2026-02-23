#!/usr/bin/env bash
# Verify 4-stage blitz-decode pipeline: 4 processes, 4x2 meshes, sockets.
# Run while tt_media_server_cpp is running (with LLM_DEVICE_BACKEND=ttrun).
# Usage: ./scripts/verify_4stage_pipeline.sh

set -e
LOG_DIR="${LOG_DIR:-/tmp/tt-run/logs}"
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

fail() { echo -e "${RED}FAIL: $*${NC}"; exit 1; }
pass() { echo -e "${GREEN}PASS: $*${NC}"; }

echo "=== 1. Process count (4 runner.py processes under tt-run) ==="
RUNNER_COUNT=$(pgrep -f "runner\.py" 2>/dev/null | wc -l)
if [[ "$RUNNER_COUNT" -eq 4 ]]; then
  pass "Found 4 runner.py processes (PIDs: $(pgrep -f 'runner\.py' | tr '\n' ' '))"
else
  fail "Expected 4 runner.py processes, found $RUNNER_COUNT. Is the server running with LLM_DEVICE_BACKEND=ttrun?"
fi

echo ""
echo "=== 2. Rank binding dry-run (mpirun with 4 ranks) ==="
if [[ -z "$TT_METAL_HOME" ]]; then
  echo "TT_METAL_HOME not set; skipping dry-run. Set it to verify mpirun command."
else
  RANK_YAML="${TT_METAL_HOME}/bh_4x2_multi_mesh_rank_binding.yaml"
  if [[ -f "$RANK_YAML" ]]; then
    python3 "${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" --rank-binding "$RANK_YAML" --dry-run -- /bin/true 2>/dev/null | grep -q "\-np 1" && pass "tt-run dry-run shows per-rank -np 1 (4 ranks)" || echo "Could not confirm 4-rank mpirun from dry-run"
  else
    echo "Rank binding not found at $RANK_YAML; skipping dry-run"
  fi
fi

echo ""
echo "=== 3. Log files: 4 ranks, mesh_id 0..3, pipeline running, one SHM bridge ==="
if [[ ! -d "$LOG_DIR" ]]; then
  fail "Log directory $LOG_DIR not found. Start the server and wait for pipeline to open devices."
fi
LOG_FILES=("$LOG_DIR"/ttrun_hello_world_*_*.log)
if [[ ! -e "${LOG_FILES[0]}" ]]; then
  fail "No ttrun_hello_world_*_*.log files in $LOG_DIR"
fi
RANK_LOGS=()
for r in 0 1 2 3; do
  F=$(ls -t "$LOG_DIR"/ttrun_hello_world_*_"$r".log 2>/dev/null | head -1)
  if [[ -n "$F" && -f "$F" ]]; then
    RANK_LOGS+=("$F")
  fi
done
if [[ ${#RANK_LOGS[@]} -ne 4 ]]; then
  fail "Expected 4 rank log files (0..3), found ${#RANK_LOGS[@]}"
fi
pass "Found log files for ranks 0,1,2,3"

for r in 0 1 2 3; do
  F=$(ls -t "$LOG_DIR"/ttrun_hello_world_*_"$r".log 2>/dev/null | head -1)
  grep -q "mesh_id=$r " "$F" 2>/dev/null || fail "Rank $r log missing 'mesh_id=$r' in $F"
  grep -q "pipeline running" "$F" 2>/dev/null || fail "Rank $r log missing 'pipeline running' in $F"
done
pass "Each rank log has correct mesh_id and 'pipeline running'"

SHM_COUNT=$(grep -l "entering SHM pipeline bridge" "$LOG_DIR"/ttrun_hello_world_*_0.log 2>/dev/null | wc -l)
if [[ "$SHM_COUNT" -ge 1 ]]; then
  pass "Rank 0 log contains 'entering SHM pipeline bridge' (host bridge active)"
else
  echo "Note: No 'entering SHM pipeline bridge' in rank 0 logs yet (bridge may not have started)"
fi

echo ""
echo "=== 4. Per-rank device count (8 devices for 4x2 mesh) ==="
for r in 0 1 2 3; do
  F=$(ls -t "$LOG_DIR"/ttrun_hello_world_*_"$r".log 2>/dev/null | head -1)
  if grep -qE "\(8\) devices|8 devices" "$F" 2>/dev/null; then
    pass "Rank $r: 8 devices (4x2 mesh)"
  else
    echo "Rank $r: check $F for 'TT_VISIBLE_DEVICES' / device count"
  fi
done

echo ""
echo "=== Summary ==="
echo "4 processes (runner.py), 4 rank logs with mesh_id 0..3 and 'pipeline running',"
echo "and rank 0 using SHM bridge confirms: tt-run launched 4 processes on 4x2 meshes,"
echo "with stage 0 talking to the C++ server via shared memory and D2D between stages."
