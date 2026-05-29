#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Compare Dynamo request-plane throughput: TCP vs HTTP.
#
# Runs vllm bench serve against a fresh frontend + cpp_server backend for
# each plane, then prints a side-by-side summary. Uses mock_pipeline by
# default so results reflect request-plane + server overhead, not device I/O.
#
# Stop any other local Dynamo stack (e.g. ./scripts/start_dynamo.sh) before
# running so etcd and BlazeRunner are not shared with this harness.
#
# Usage:
#   ./scripts/bench_dynamo_request_planes.sh
#   NUM_PROMPTS=500 MAX_CONCURRENCY=32 ./scripts/bench_dynamo_request_planes.sh
#   DYN_DISCOVERY_BACKEND=etcd ETCD_ENDPOINTS=http://127.0.0.1:2379 ./scripts/...
#
# Requires: built tt_media_server_cpp, DYN_VENV with dynamo.frontend, vllm in
# PATH or BENCH_VENV with vllm installed (see install_bench_client.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
BIN="${BUILD_DIR}/tt_media_server_cpp"
DEFAULT_DYN_VENV="${SCRIPT_DIR}/../../../dynamo-mock-backend/.venv"

# --- bench knobs ------------------------------------------------------------
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-128}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-128}"
# Use high ports by default to avoid colliding with a dev ./scripts/start_dynamo.sh.
HTTP_PORT="${HTTP_PORT:-18090}"
SERVER_PORT="${SERVER_PORT:-18001}"
DYN_HTTP_RPC_PORT="${DYN_HTTP_RPC_PORT:-18888}"
FRONTEND_MODEL_NAME="${FRONTEND_MODEL_NAME:-deepseek-ai/DeepSeek-R1-0528}"
MODEL_NAME="${MODEL_NAME:-}"  # bench model id; defaults to FRONTEND_MODEL_NAME
DEVICE_IDS="${DEVICE_IDS:-(0)}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/dynamo_plane_bench_$(date +%Y%m%d_%H%M%S)}"
TT_LOG_LEVEL="${TT_LOG_LEVEL:-warn}"
PLANES="${PLANES:-tcp http}"
# vllm bench is client-only; no GPU required on the bench host.
VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-empty}"

DYN_VENV="${DYN_VENV:-${DEFAULT_DYN_VENV}}"
# Prefer a dedicated bench venv (see scripts/ci/install_bench_client.sh).
BENCH_VENV="${BENCH_VENV:-/tmp/bench_vllm}"
# cpp_server registers workers via etcd only (see discovery.cpp).
DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-etcd}"
DYN_FILE_STORE="${DYN_FILE_STORE:-/tmp/dynamo_plane_bench_store}"
ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"
LLM_DEVICE_BACKEND="${LLM_DEVICE_BACKEND:-mock_pipeline}"
DYN_HTTP_RPC_ROOT_PATH="${DYN_HTTP_RPC_ROOT_PATH:-/v1/rpc}"
DYN_HTTP_RPC_HOST="${DYN_HTTP_RPC_HOST:-127.0.0.1}"
DYN_EVENT_PLANE="${DYN_EVENT_PLANE:-zmq}"

FRONTEND_PID=""
BACKEND_PID=""

usage() {
  cat <<EOF
Usage: $0

Environment (common overrides):
  NUM_PROMPTS=1000          Total requests per plane
  MAX_CONCURRENCY=64        Client concurrency
  RANDOM_INPUT_LEN=128    Synthetic prompt length
  RANDOM_OUTPUT_LEN=128     Synthetic completion length
  PLANES="tcp http"         Space-separated planes to benchmark
  OUTPUT_DIR=...            Where JSON results and logs are written
  DYN_VENV=...              venv with dynamo.frontend
  BENCH_VENV=...            venv with vllm (optional; else PATH)
  DYN_DISCOVERY_BACKEND=file|etcd
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

vllm_cmd() {
  if [[ -n "${BENCH_VENV}" && -x "${BENCH_VENV}/bin/vllm" ]]; then
    echo "${BENCH_VENV}/bin/vllm"
  elif command -v vllm >/dev/null 2>&1; then
    echo vllm
  else
    echo ""
  fi
}

require_prereqs() {
  if [[ ! -x "${BIN}" ]]; then
    echo "Building cpp_server..." >&2
    "${SCRIPT_DIR}/build.sh"
  fi
  if [[ ! -d "${DYN_VENV}" ]]; then
    echo "DYN_VENV not found: ${DYN_VENV}" >&2
    exit 1
  fi
  if [[ ! -f "${DYN_VENV}/bin/python3" ]]; then
    echo "Missing ${DYN_VENV}/bin/python3" >&2
    exit 1
  fi
  local vllm
  vllm="$(vllm_cmd)"
  if [[ -z "${vllm}" ]]; then
    echo "vllm not found. Install via:" >&2
    echo "  ${SCRIPT_DIR}/scripts/ci/install_bench_client.sh --venv ~/.venvs/bench" >&2
    echo "  export BENCH_VENV=~/.venvs/bench" >&2
    exit 1
  fi
  if [[ "${DYN_DISCOVERY_BACKEND}" == "etcd" ]]; then
    local host_port="${ETCD_ENDPOINTS#http://}"
    host_port="${host_port%%/*}"
    if ! (echo >"/dev/tcp/${host_port/:/\/}") 2>/dev/null; then
      echo "etcd not reachable at ${ETCD_ENDPOINTS}" >&2
      exit 1
    fi
  fi
}

cleanup_etcd_workers() {
  if [[ "${DYN_DISCOVERY_BACKEND}" != "etcd" ]] || ! command -v etcdctl >/dev/null 2>&1; then
    return 0
  fi
  echo "Clearing stale etcd worker keys under default/backend/generate ..." >&2
  etcdctl --endpoints="${ETCD_ENDPOINTS}" del --prefix \
    "v1/instances/default/backend/generate/" >/dev/null 2>&1 || true
  etcdctl --endpoints="${ETCD_ENDPOINTS}" del --prefix \
    "v1/mdc/default/backend/generate/" >/dev/null 2>&1 || true
}

stop_stack() {
  [[ -n "${FRONTEND_PID}" ]] && kill "${FRONTEND_PID}" 2>/dev/null || true
  [[ -n "${BACKEND_PID}" ]] && kill "${BACKEND_PID}" 2>/dev/null || true
  FRONTEND_PID=""
  BACKEND_PID=""
  wait 2>/dev/null || true
  sleep 1
}

wait_http() {
  local url="$1"
  local label="$2"
  local logfile="$3"
  for _ in $(seq 1 60); do
    if curl -sf "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "ERROR: ${label} not ready (${url})" >&2
  tail -40 "${logfile}" >&2 || true
  return 1
}

model_path_for_backend() {
  case "${LLM_DEVICE_BACKEND}" in
    llama*|*llama*) echo "${SCRIPT_DIR}/tokenizers/meta-llama/Llama-3.1-8B-Instruct" ;;
    *) echo "${SCRIPT_DIR}/tokenizers/deepseek-ai/DeepSeek-R1-0528" ;;
  esac
}

start_stack() {
  local plane="$1"
  local plane_dir="${OUTPUT_DIR}/${plane}"
  mkdir -p "${plane_dir}"

  export DYN_REQUEST_PLANE="${plane}"
  export DYNAMO_REQUEST_PLANE="${plane}"
  export DYNAMO_ENDPOINT_ENABLED=1
  export LLM_DEVICE_BACKEND
  export DYN_DISCOVERY_BACKEND
  export DYNAMO_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND}"
  export ETCD_ENDPOINTS
  export DYNAMO_ETCD_ENDPOINTS="${ETCD_ENDPOINTS}"
  export DYN_HTTP_RPC_PORT DYN_HTTP_RPC_ROOT_PATH DYN_HTTP_RPC_HOST
  export DYN_EVENT_PLANE DEVICE_IDS
  export DYNAMO_NAMESPACE="${DYNAMO_NAMESPACE:-default}"
  export DYNAMO_COMPONENT="${DYNAMO_COMPONENT:-backend}"
  export DYNAMO_ENDPOINT_NAME="${DYNAMO_ENDPOINT_NAME:-generate}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-dynamo-bypass}"
  export TT_LOG_LEVEL

  if [[ "${DYN_DISCOVERY_BACKEND}" == "file" ]]; then
    export DYN_FILE_STORE="${DYN_FILE_STORE}_${plane}"
    export DYNAMO_DISCOVERY_PATH="${DYN_FILE_STORE}"
    rm -rf "${DYN_FILE_STORE}"
  fi

  local model_path
  model_path="$(model_path_for_backend)"

  echo ">>> Starting stack (plane=${plane})..." >&2
  "${DYN_VENV}/bin/python3" -m dynamo.frontend \
    --http-port "${HTTP_PORT}" \
    --model-name "${FRONTEND_MODEL_NAME}" \
    --model-path "${model_path}" \
    >"${plane_dir}/frontend.log" 2>&1 &
  FRONTEND_PID=$!
  sleep 2

  TT_LOG_LEVEL="${TT_LOG_LEVEL}" "${BIN}" -p "${SERVER_PORT}" \
    >"${plane_dir}/backend.log" 2>&1 &
  BACKEND_PID=$!

  wait_http "http://127.0.0.1:${HTTP_PORT}/v1/models" "frontend" \
    "${plane_dir}/frontend.log"
  if ! wait_http "http://127.0.0.1:${SERVER_PORT}/tt-liveness" "backend" \
    "${plane_dir}/backend.log"; then
    return 1
  fi
  # model_ready flips after tokenizer load; liveness alone is not enough.
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${SERVER_PORT}/tt-liveness" 2>/dev/null \
        | grep -q '"model_ready":true'; then
      return 0
    fi
    sleep 2
  done
  echo "ERROR: backend model not ready" >&2
  tail -40 "${plane_dir}/backend.log" >&2 || true
  return 1
}

resolve_model_name() {
  if [[ -z "${MODEL_NAME}" ]]; then
    MODEL_NAME="${FRONTEND_MODEL_NAME}"
  fi
  echo "Using model: ${MODEL_NAME}" >&2
}

wait_backend_discovered() {
  local plane_dir="$1"
  if [[ "${DYN_DISCOVERY_BACKEND}" == "etcd" ]]; then
    if ! command -v etcdctl >/dev/null 2>&1; then
      sleep 5
      return 0
    fi
    for _ in $(seq 1 60); do
      local count
      count="$(etcdctl --endpoints="${ETCD_ENDPOINTS}" get --prefix --keys-only \
        "v1/instances/default/backend/generate/" 2>/dev/null \
        | grep -c "v1/instances/default/backend/generate/" || true)"
      if [[ "${count}" -ge 1 ]]; then
        return 0
      fi
      sleep 1
    done
    echo "ERROR: backend not registered in etcd" >&2
    tail -40 "${plane_dir}/backend.log" >&2 || true
    return 1
  fi
  sleep 5
}

smoke_request() {
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' \
    "http://127.0.0.1:${HTTP_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${OPENAI_API_KEY}" \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4,\"stream\":false}")"
  if [[ "${code}" != "200" ]]; then
    echo "ERROR: smoke chat completion returned HTTP ${code}" >&2
    return 1
  fi
}

run_bench() {
  local plane="$1"
  local plane_dir="${OUTPUT_DIR}/${plane}"
  local result="${plane_dir}/vllm_bench.json"
  local vllm
  vllm="$(vllm_cmd)"

  echo ">>> vllm bench (plane=${plane}, prompts=${NUM_PROMPTS}, concurrency=${MAX_CONCURRENCY})..." >&2
  export VLLM_TARGET_DEVICE
  if ! (cd "${plane_dir}" && "${vllm}" bench serve \
    --model "${MODEL_NAME}" \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --base-url "http://127.0.0.1:${HTTP_PORT}" \
    --dataset-name random \
    --random-input-len "${RANDOM_INPUT_LEN}" \
    --random-output-len "${RANDOM_OUTPUT_LEN}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --save-result \
    --result-filename "vllm_bench.json" \
    >"${plane_dir}/bench.log" 2>&1); then
    echo "ERROR: vllm bench failed (plane=${plane})" >&2
    return 1
  fi
  result="${plane_dir}/vllm_bench.json"
  if [[ ! -f "${result}" ]]; then
    echo "ERROR: missing ${result}" >&2
    return 1
  fi

  python3 - "${result}" "${plane}" <<'PY'
import json, sys
path, plane = sys.argv[1], sys.argv[2]
with open(path) as f:
    r = json.load(f)
keys = (
    "request_throughput", "output_throughput", "total_token_throughput",
    "mean_ttft_ms", "median_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
    "completed", "failed",
)
row = {"plane": plane}
for k in keys:
    row[k] = r.get(k)
print(json.dumps(row))
if not row.get("completed"):
    raise SystemExit("no successful requests")
PY
}

print_summary() {
  python3 - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for p in sorted(out.iterdir()):
    meta = p / "metrics.json"
    if meta.is_file():
        rows.append(json.loads(meta.read_text()))
if not rows:
    print("No results found.")
    raise SystemExit(1)

def col(k, w=14):
    return k[:w].ljust(w)

hdr = ["plane", "req_tput", "out_tput", "tok_tput", "mean_ttft", "mean_tpot", "completed", "failed"]
print()
print("Dynamo request-plane comparison")
print("=" * 96)
print("  ".join(col(h) for h in hdr))
print("-" * 96)
for r in rows:
    print("  ".join(
        col(str(r.get(h, "")))
        for h in hdr
    ))
print()
if len(rows) == 2:
    a, b = rows[0], rows[1]
    def pct(new, old):
        if old in (None, 0, 0.0) or new is None:
            return "n/a"
        return f"{100.0 * (float(new) - float(old)) / float(old):+.1f}%"
    print(f"Δ request_throughput ({b['plane']} vs {a['plane']}): {pct(b.get('request_throughput'), a.get('request_throughput'))}")
    print(f"Δ mean_ttft_ms          ({b['plane']} vs {a['plane']}): {pct(b.get('mean_ttft_ms'), a.get('mean_ttft_ms'))}")
    print(f"Δ mean_tpot_ms          ({b['plane']} vs {a['plane']}): {pct(b.get('mean_tpot_ms'), a.get('mean_tpot_ms'))}")
print(f"\nArtifacts: {out}")
PY
}

# --- main -------------------------------------------------------------------
require_prereqs
mkdir -p "${OUTPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Planes: ${PLANES}"
echo ""

trap 'stop_stack' EXIT INT TERM

for plane in ${PLANES}; do
  stop_stack
  cleanup_etcd_workers
  sleep 1
  start_stack "${plane}" || exit 1
  wait_backend_discovered "${OUTPUT_DIR}/${plane}" || exit 1
  resolve_model_name
  smoke_request || exit 1
  metrics="$(run_bench "${plane}")" || exit 1
  echo "${metrics}" >"${OUTPUT_DIR}/${plane}/metrics.json"
  stop_stack
  sleep 3
done

print_summary
