#!/bin/bash
# Invoked automatically via TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE when
# tt-metal's TT_METAL_OPERATION_TIMEOUT_SECONDS watchdog detects a stuck
# device op (see #4521 -- device-read stall in FDMeshCommandQueue::
# read_completion_queue). Dumps a full gdb thread backtrace of the live
# EngineCore process so the hang signature is captured automatically, without
# needing to notice + manually attach gdb after the fact.
set -u
LOG_DIR="$(dirname "$0")"
STAMP=$(date +%Y%m%d-%H%M%S)
for pid in $(pgrep -f "VLLM::EngineCore"); do
  OUT="$LOG_DIR/hang_gdb_${pid}_${STAMP}.log"
  echo "[dump_hang_gdb] dumping EngineCore pid=$pid -> $OUT"
  timeout 60 gdb -p "$pid" -batch -ex "thread apply all bt" > "$OUT" 2>&1
done
