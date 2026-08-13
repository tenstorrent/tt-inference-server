#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# Small helper to stream a chat completion against a running tt_media_server_cpp.
#
# Configuration via environment variables:
#   OPENAI_API_BASE  full server base URL (scheme://host[:port]); overrides HOST/PORT
#   HOST             server host  (default: localhost)
#   PORT             server port  (default: 8000)
#   MODEL            model name   (default: deepseek-ai/DeepSeek-R1-0528)
#
# Usage (flags accepted in any position):
#   ./chat.sh "Hello, how are you?"
#   PORT=8001 ./chat.sh "Tell me a joke."
#   HOST=127.0.0.1 PORT=8001 ./chat.sh "Summarize the moon landing." -n 64
#
# Prints assistant tokens to stdout as they arrive, then a final newline.

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
MAX_TOKENS=128
API_KEY="${OPENAI_API_KEY:-your-secret-key}"
PROMPT=""

usage() {
  cat >&2 <<EOF
Usage: $0 [-n MAX_TOKENS] "prompt text"
  -n MAX_TOKENS  max completion tokens (default: $MAX_TOKENS)

Environment variables:
  OPENAI_API_BASE  full server base URL (scheme://host[:port]); overrides HOST/PORT
  HOST             server host  (default: localhost)
  PORT             server port  (default: 8000)
  MODEL            model name   (default: deepseek-ai/DeepSeek-R1-0528)
Flags may appear before or after the prompt.
EOF
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    -n) MAX_TOKENS="$2"; shift 2 ;;
    -h|--help) usage ;;
    --) shift; PROMPT="${PROMPT:+$PROMPT }$*"; break ;;
    -*) echo "Unknown flag: $1" >&2; usage ;;
    *)  PROMPT="${PROMPT:+$PROMPT }$1"; shift ;;
  esac
done

if [ -z "$PROMPT" ]; then
  usage
fi

# Build the JSON body — jq escapes the prompt safely.
BODY=$(jq -n \
  --arg model "$MODEL" \
  --arg prompt "$PROMPT" \
  --argjson max_tokens "$MAX_TOKENS" \
  '{
    model: $model,
    messages: [{role: "user", content: $prompt}],
    max_tokens: $max_tokens,
    stream: true,
    stream_options: {include_usage: true}
  }')

# OPENAI_API_BASE (a full scheme://host[:port]) wins when set — needed to reach an
# HTTPS endpoint (e.g. an ngrok tunnel on :443); otherwise fall back to HOST:PORT.
BASE_URL="${OPENAI_API_BASE:-http://${HOST}:${PORT}}"
URL="${BASE_URL%/}/v1/chat/completions"

# Stream SSE, extract delta.content from each frame, print as it arrives.
curl -sS -N --fail-with-body \
  -X POST "$URL" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$BODY" \
| jq --raw-input --unbuffered -j '
    select(startswith("data: "))
    | sub("^data: "; "")
    | select(. != "[DONE]")
    | (fromjson? // {})
    | ((.choices[0].delta.reasoning_content // "") + (.choices[0].delta.content // ""))
  '
status=${PIPESTATUS[0]}

echo
if [ "$status" -ne 0 ]; then
  echo "[chat.sh] curl exited with status $status (is the server up at ${HOST}:${PORT}?)" >&2
  exit "$status"
fi
