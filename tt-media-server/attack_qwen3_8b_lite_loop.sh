#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Hang repro loop for attack_qwen3_8b_lite.sh, ported from
# hang_repro_llama_3_1_8b.sh. Repeatedly fires the conc=64 vllm bench serve
# burst against an ALREADY-RUNNING Qwen3-8B server (e.g. via
# launch_qwen3_8b_debug.sh), each run to its own logfile, until one run fails
# to finish within the timeout (= hang) or NUM_RUNS is reached.
#
# Usage:
#   ./attack_qwen3_8b_lite_loop.sh [num_runs] [port] [wall_timeout_secs]
# Examples:
#   ./attack_qwen3_8b_lite_loop.sh                    # 20 runs, port 8019, 60s timeout/run
#   ./attack_qwen3_8b_lite_loop.sh 40 8019 90          # 40 runs, 90s timeout/run

NUM_RUNS=${1:-10}
PORT=${2:-8019}
RUN_TIMEOUT=${3:-30}

SUFFIX=qwen3_8b_lite_$(date +%Y%m%d_%H%M%S)

for i in $(seq 1 "$NUM_RUNS"); do
  LOGFILE="attack_${SUFFIX}_${i}.log"
  echo "=== run $i/$NUM_RUNS -> $LOGFILE ==="
  timeout "$RUN_TIMEOUT" ./attack_qwen3_8b_lite.sh "$LOGFILE" "$PORT"
  RC=$?
  if [ "$RC" -eq 124 ]; then
    echo "run $i ($LOGFILE) HANG -- timeout ($RUN_TIMEOUT s) fired, attack did not complete"
    break
  elif grep -q "Successful requests:.*64" "$LOGFILE" && ! grep -q "Failed requests:\s*[1-9]" "$LOGFILE"; then
    echo "run $i ($LOGFILE) PASS"
  else
    echo "run $i ($LOGFILE) HANG -- no clean 64/64 success found in log"
    break
  fi
done
