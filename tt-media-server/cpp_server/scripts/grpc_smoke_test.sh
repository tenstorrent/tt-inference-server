#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Quick gRPC smoke test for tt_media_server_cpp (requires --grpc build).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_SERVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BINARY="${CPP_SERVER_DIR}/build/tt_media_server_cpp"
HTTP_PORT="${HTTP_PORT:-18000}"
GRPC_PORT="${GRPC_PORT:-50051}"
GRPC_ADDR="127.0.0.1:${GRPC_PORT}"
PROTO_FILE="${CPP_SERVER_DIR}/protos/inference.proto"
GRPCURL_COMMON=(
  -plaintext
  -import-path "${CPP_SERVER_DIR}/protos"
  -proto "$(basename "${PROTO_FILE}")"
)
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ ! -x "${BINARY}" ]]; then
  echo "ERROR: ${BINARY} not found. Build with: ./build.sh --grpc"
  exit 1
fi

if ! command -v grpcurl >/dev/null 2>&1; then
  echo "ERROR: grpcurl not found. Install: go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest"
  echo "       or: sudo apt install grpcurl (if available)"
  exit 1
fi

echo "Starting server (HTTP :${HTTP_PORT}, gRPC :${GRPC_PORT})..."
GRPC_LISTEN="0.0.0.0:${GRPC_PORT}" \
  LLM_DEVICE_BACKEND=mock_pipeline \
  "${BINARY}" -p "${HTTP_PORT}" &
SERVER_PID=$!

for _ in $(seq 1 60); do
  if grpcurl "${GRPCURL_COMMON[@]}" "${GRPC_ADDR}" inference.Inference/Health >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

echo "--- Health ---"
grpcurl "${GRPCURL_COMMON[@]}" "${GRPC_ADDR}" inference.Inference/Health

echo "--- Generate (streaming, 3 prompt tokens) ---"
grpcurl "${GRPCURL_COMMON[@]}" -d '{
  "model": "deepseek-ai/DeepSeek-R1-0528",
  "token_ids": [1, 2, 3],
  "stop_conditions": { "max_tokens": 4 }
}' "${GRPC_ADDR}" inference.Inference/Generate

echo ""
echo "gRPC smoke test OK"
