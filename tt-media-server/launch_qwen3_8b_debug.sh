#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# #4521 debug wrapper around launch_qwen3_8b_ci.sh: same production config, plus
# tt-metal hang self-detection + tt-mlir per-op/per-program tracing and (optionally)
# forced per-op device sync, using the debug-instrumented libTTMLIRRuntime.so built
# during the 2026-07-15 overnight investigation (see
# tt-inference-server/HANDOFF_4521_OVERNIGHT_2026-07-15.md).
#
# Requires the instrumented lib already rebuilt + installed at
# ~/tt-xla/third_party/tt-mlir/install/lib/libTTMLIRRuntime.so (LOG_DEBUG->LOG_INFO
# for per-op/program-start/program-finish logging, syncAfterOpIfNeeded() un-gated
# from TT_RUNTIME_DEBUG). If you rebuild tt-mlir from a clean checkout, re-apply
# those changes first or this will just behave like a normal (non-instrumented) run.
#
# NOTE: TT_RUNTIME_SYNC_AFTER_OP=1 is incompatible with ENABLE_TRACE=true (tt-metal
# throws "Event Synchronization is not supported during trace capture"). This script
# forces ENABLE_TRACE=false whenever sync-after-op is on (default: on). Warmup is
# MUCH slower with sync-after-op on (~650s+ just for the single-request warmup,
# hundreds of thousands of extra log lines) since every op round-trips the device.
#
# Run (from the tt-xla venv, pinned to chip 0):
#   cd /home/kmabee/tt-inference-server/tt-media-server && ./launch_qwen3_8b_debug.sh
# (this script cd's into ~/tt-xla and sources venv/activate itself, so you do not
# need to do that part manually first.)
#
# Toggle any DEBUG_* knob via env, e.g. to skip the (slow) per-op sync and just get
# the fast/reliable repro + auto-gdb-on-timeout:
#   DEBUG_SYNC_AFTER_OP=0 ./launch_qwen3_8b_debug.sh

DEBUG_OP_TIMEOUT_SECS=${DEBUG_OP_TIMEOUT_SECS:-10}
DEBUG_RUNTIME_LOG_LEVEL=${DEBUG_RUNTIME_LOG_LEVEL:-INFO}
DEBUG_SYNC_AFTER_OP=${DEBUG_SYNC_AFTER_OP:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd /home/kmabee/tt-xla && source venv/activate
cd "$SCRIPT_DIR"

export TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-$DEBUG_OP_TIMEOUT_SECS}
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE=${TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE:-"$SCRIPT_DIR/dump_hang_gdb.sh"}
export TTMLIR_RUNTIME_LOGGER_LEVEL=${TTMLIR_RUNTIME_LOGGER_LEVEL:-$DEBUG_RUNTIME_LOG_LEVEL}

export NUM_HIDDEN_LAYERS=1
export CPU_SAMPLING=true

if [ "$DEBUG_SYNC_AFTER_OP" = "1" ]; then
  export TT_RUNTIME_SYNC_AFTER_OP=1
  export ENABLE_TRACE=false
  echo "[launch_qwen3_8b_debug] sync-after-op ON -> forcing ENABLE_TRACE=false (incompatible with trace capture)"
else
  echo "[launch_qwen3_8b_debug] sync-after-op OFF (DEBUG_SYNC_AFTER_OP=0) -> ENABLE_TRACE left at script default"
fi

echo "[launch_qwen3_8b_debug] CPU_SAMPLING=$CPU_SAMPLING"
echo "[launch_qwen3_8b_debug] TT_METAL_OPERATION_TIMEOUT_SECONDS=$TT_METAL_OPERATION_TIMEOUT_SECONDS"
echo "[launch_qwen3_8b_debug] TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE=$TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"
echo "[launch_qwen3_8b_debug] TTMLIR_RUNTIME_LOGGER_LEVEL=$TTMLIR_RUNTIME_LOGGER_LEVEL"
echo "[launch_qwen3_8b_debug] TT_RUNTIME_SYNC_AFTER_OP=${TT_RUNTIME_SYNC_AFTER_OP:-<unset>}"
echo "[launch_qwen3_8b_debug] ENABLE_TRACE=${ENABLE_TRACE:-<script default: true>}"

exec ./launch_qwen3_8b_ci.sh
