#!/usr/bin/env bash
# Run the e2e release flow for Qwen3-8B against an ALREADY-RUNNING uvicorn server
# (e.g. tt-media-server/launch_qwen3_8b.sh on port 8004), via run.py --workflow
# release: evals -> benchmarks -> spec_tests -> tests -> reports.
#
# Does NOT launch its own server (no --docker-server / --local-server); it targets
# the live server with --server-url + --service-port. Start the server first.
# Waits for the server to finish warmup (and aborts if the EngineCore dies during
# warmup, e.g. an OOM at trace-capture) before launching the release flow.
#
# Override via env: HOST, SERVICE_PORT, KEY, LIMIT_MODE, MODEL_SPECS_ENV.
set -uo pipefail
cd "$(dirname "$0")"

MODEL="Qwen3-8B"
HOST="${HOST:-127.0.0.1}"
SERVICE_PORT="${SERVICE_PORT:-8004}"
KEY="${KEY:-your-secret-key}"                 # must match the running server's API_KEY
LIMIT_MODE="${LIMIT_MODE:-ci-nightly}"        # ci-nightly | ci-long | ci-commit | smoke-test
export MODEL_SPECS_ENV="${MODEL_SPECS_ENV:-dev}"
# run.py needs yaml/pydantic/etc. The bare `python` isn't always on PATH in
# non-login shells, so resolve an interpreter that actually has run.py's deps.
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for cand in /opt/ttmlir-toolchain/venv/bin/python python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import yaml" >/dev/null 2>&1; then
      PYTHON="$cand"; break
    fi
  done
fi
[[ -z "$PYTHON" ]] && { echo "ERROR: no python with run.py deps found (set PYTHON=/path/to/python)"; exit 1; }
LOG="local_release_qwen3_8b_p${SERVICE_PORT}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

echo "=== WAIT-FOR-WARMUP START: $(date -u) (model=$MODEL @ ${HOST}:${SERVICE_PORT}) ==="
ready=0
for i in $(seq 1 60); do   # up to ~30 min
  resp=$(curl -s -m 90 "http://${HOST}:${SERVICE_PORT}/v1/chat/completions" \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"hi"}],"max_tokens":3}' 2>/dev/null)
  if echo "$resp" | grep -q '"choices"'; then echo "[poll $i @ $(date -u +%H:%M:%S)] READY"; ready=1; break; fi
  # early-abort: parent uvicorn up but no EngineCore => warmup died (e.g. OOM at trace-capture)
  if pgrep -f "uvicorn main:app.*${SERVICE_PORT}" >/dev/null 2>&1 && ! pgrep -f "EngineCore" >/dev/null 2>&1; then
    echo "[poll $i @ $(date -u +%H:%M:%S)] ENGINE_DEAD — warmup failed"; ready=dead; break
  fi
  echo "[poll $i @ $(date -u +%H:%M:%S)] ${resp:0:90}"; sleep 30
done

if [ "$ready" != "1" ]; then
  echo "=== ABORT: server not ready (ready=$ready) at $(date -u). Not launching release flow. ==="
  exit 1
fi

echo "=== RELEASE RUN START: $(date -u) | limit-samples-mode=$LIMIT_MODE specs=$MODEL_SPECS_ENV ==="
SECONDS=0
echo "Using python: $PYTHON"
VLLM_API_KEY="$KEY" OPENAI_API_KEY="$KEY" \
"$PYTHON" run.py \
  --model "$MODEL" \
  --device p150 \
  --impl forge-vllm-plugin \
  --workflow release \
  --server-url "http://${HOST}" \
  --service-port "$SERVICE_PORT" \
  --limit-samples-mode "$LIMIT_MODE"
rc=$?
echo "=== RELEASE RUN END: $(date -u) | elapsed ${SECONDS}s (~$((SECONDS/60))m) | exit=$rc ==="
exit $rc
