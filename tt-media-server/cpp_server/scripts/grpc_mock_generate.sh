#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Call inference.Inference/Generate on a running server (mock backend).
#
# The gRPC API is token-in / token-out. Mock returns generated token IDs in a
# stream (not decoded text). For natural-language chat, use ./chat.sh on HTTP.
#
# Usage (server must already be running with GRPC_LISTEN set):
#   GRPC_LISTEN=0.0.0.0:50051 LLM_DEVICE_BACKEND=mock_pipeline ./build/tt_media_server_cpp -p 8000
#
#   ./scripts/grpc_mock_generate.sh
#   ./scripts/grpc_mock_generate.sh --max-tokens 8 --token-ids 100,101,102
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_SERVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GRPC_ADDR="${GRPC_ADDR:-127.0.0.1:50051}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
MAX_TOKENS=4
TOKEN_IDS="1,2,3"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --addr HOST:PORT     gRPC address (default: ${GRPC_ADDR})
  --model NAME         model field in GenerateRequest (default: ${MODEL})
  --max-tokens N       max_tokens (default: ${MAX_TOKENS})
  --token-ids LIST     comma-separated prompt token ids (default: ${TOKEN_IDS})
  -h, --help           show this help

Service/method: inference.Inference/Generate
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --addr) GRPC_ADDR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --token-ids) TOKEN_IDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v grpcurl >/dev/null 2>&1; then
  echo "ERROR: grpcurl not found" >&2
  exit 1
fi

# Build JSON array for token_ids
IFS=',' read -ra IDS <<< "${TOKEN_IDS}"
JSON_IDS=""
for id in "${IDS[@]}"; do
  id="${id//[[:space:]]/}"
  [[ -n "${id}" ]] || continue
  JSON_IDS="${JSON_IDS}${JSON_IDS:+,}${id}"
done

echo "gRPC ${GRPC_ADDR} inference.Inference/Generate"
echo "  model=${MODEL} token_ids=[${JSON_IDS}] stop_conditions.max_tokens=${MAX_TOKENS}"
echo "--- stream ---"

grpcurl -plaintext \
  -import-path "${CPP_SERVER_DIR}/protos" \
  -proto inference.proto \
  -d "{\"model\":\"${MODEL}\",\"token_ids\":[${JSON_IDS}],\"stop_conditions\":{\"max_tokens\":${MAX_TOKENS}}}" \
  "${GRPC_ADDR}" inference.Inference/Generate

echo "--- done ---"
echo ""
echo "Note: mock streams token IDs. For readable chat text on HTTP:"
echo "  ./chat.sh -p 8000 \"Your message here\""
