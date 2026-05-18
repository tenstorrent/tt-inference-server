#!/usr/bin/env bash
# Small helper to stream a chat completion against a running tt_media_server_cpp.
#
# Usage (flags accepted in any position):
#   ./chat.sh "Hello, how are you?"
#   ./chat.sh -p 8001 "Tell me a joke."
#   ./chat.sh "Summarize the moon landing." -p 8001 -n 64
#
# Prints assistant tokens to stdout as they arrive, then a final newline.

PORT=8000
MAX_TOKENS=128
API_KEY="${OPENAI_API_KEY:-your-secret-key}"
PROMPT=""

usage() {
  cat >&2 <<EOF
Usage: $0 [-p PORT] [-n MAX_TOKENS] "prompt text"
  -p PORT        server port (default: $PORT)
  -n MAX_TOKENS  max completion tokens (default: $MAX_TOKENS)
Flags may appear before or after the prompt.
EOF
  exit 1
}

# Manual arg loop so -p / -n work in any position.
while [ $# -gt 0 ]; do
  case "$1" in
    -p) PORT="$2"; shift 2 ;;
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
  --arg prompt "$PROMPT" \
  --argjson max_tokens "$MAX_TOKENS" \
  '{
    model: "deepseek-ai/DeepSeek-R1-0528",
    messages: [{role: "user", content: $prompt}],
    max_tokens: $max_tokens,
    stream: true,
    stream_options: {include_usage: true}
  }')

URL="http://localhost:${PORT}/v1/chat/completions"

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
    | .choices[0].delta.content // ""
  '
status=${PIPESTATUS[0]}

echo
if [ "$status" -ne 0 ]; then
  echo "[chat.sh] curl exited with status $status (is the server up on port $PORT?)" >&2
  exit "$status"
fi
