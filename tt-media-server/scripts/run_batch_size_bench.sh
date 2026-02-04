#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Runs the media server, waits for warmup, runs vllm bench, then collects
# BGE forward logs and computes average batch_size.
# Output: output.log (matching log lines), and prints average batch_size.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
WORK_DIR="$(pwd)"

READY_MARKER="All devices are warmed up and ready"
BGE_LOG_PATTERN="BGE forward: processing batch_size="
SERVER_LOG="$WORK_DIR/server_raw.log"
OUTPUT_LOG="$WORK_DIR/output.log"
BENCH_TIMEOUT=600
READY_TIMEOUT=600

export VLLM_USE_V1=1
export ARCH_NAME=wormhole_b0
export ENVIRONMENT=development
export TT_METAL_HOME=/home/ubuntu/tt-metal
export PYTHONPATH=/home/ubuntu/tt-metal
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export HF_TOKEN="${HF_TOKEN:-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}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-/home/ubuntu/tt-metal/python_env/bin/python}"
OPENAI_API_KEY="${OPENAI_API_KEY:-your-secret-key}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    # Avoid blocking: give process up to 10s to exit, then SIGKILL
    i=0
    while [[ $i -lt 10 ]] && kill -0 "$SERVER_PID" 2>/dev/null; do
      sleep 1
      i=$((i + 1))
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting server (logs -> $SERVER_LOG)..."
"$PYTHON" -m uvicorn main:app --port 8000 --lifespan on --workers 1 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for '$READY_MARKER' (timeout ${READY_TIMEOUT}s)..."
elapsed=0
while ! grep -q "$READY_MARKER" "$SERVER_LOG" 2>/dev/null; do
  sleep 5
  elapsed=$((elapsed + 5))
  if [[ $elapsed -ge $READY_TIMEOUT ]]; then
    echo "Timeout waiting for warmup. Last lines of server log:"
    tail -n 30 "$SERVER_LOG"
    exit 1
  fi
done
echo "Server ready."

echo "Running hey benchmark..."
timeout "$BENCH_TIMEOUT" hey -n 1 -c 1 -m POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{"model":"BAAI/bge-large-en-v1.5","input":"test text here test text here test text here test text here test text here test text here test text here test text here"}' \
  http://localhost:8000/v1/embeddings || true


echo "Benchmark finished. Allowing 5s for final logs..."
sleep 5

cleanup
trap - EXIT

if [[ ! -f "$SERVER_LOG" ]]; then
  echo "Error: server log not found at $SERVER_LOG"
  exit 1
fi

echo "Collecting BGE forward logs into $OUTPUT_LOG..."
grep "$BGE_LOG_PATTERN" "$SERVER_LOG" > "$OUTPUT_LOG" || true

if [[ ! -s "$OUTPUT_LOG" ]]; then
  echo "No lines matching '$BGE_LOG_PATTERN' found. Check $SERVER_LOG."
  rm -f "$SERVER_LOG"
  exit 1
fi

count=$(wc -l < "$OUTPUT_LOG")
sum=$(awk -F'batch_size=' '{ sub(/,.*/, "", $2); sum += $2 } END { print sum+0 }' "$OUTPUT_LOG")
average=$(awk "BEGIN { printf \"%.4f\", $sum / $count }")
echo "Total BGE forward lines: $count"
echo "Sum of batch_size: $sum"
echo "Average batch_size: $average"
echo ""
echo "Batch size distribution:"
awk -F'batch_size=' '
  { sub(/,.*/, "", $2); count[$2]++ }
  END {
    for (b = 1; b <= 8; b++) printf "  batch_size=%d: %d\n", b, count[b]+0
  }
' "$OUTPUT_LOG"

pkill -9 -f "uvicorn"
