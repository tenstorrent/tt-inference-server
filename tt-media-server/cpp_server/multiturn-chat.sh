#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

PORT=8000
MAX_TOKENS=128
API_KEY="${OPENAI_API_KEY:-your-secret-key}"
PROMPT=""

usage() {
  cat >&2 <<EOF
Usage: $0 [-p PORT] [-n MAX_TOKENS] "initial prompt text"
  -p PORT        server port (default: $PORT)
  -n MAX_TOKENS  max completion tokens per turn (default: $MAX_TOKENS)
Flags may appear before or after the initial prompt.
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

URL="http://localhost:${PORT}/v1/chat/completions"

# Stateful chat history. Starts with the initial user prompt; each turn we
# append the streamed assistant reply, then the next user line.
MESSAGES_JSON=$(jq -n --arg prompt "$PROMPT" '[{role: "user", content: $prompt}]')

echo "Starting chat session. Type 'exit' or 'quit' to end."
echo "Press Ctrl+C during the assistant's reply to cancel that turn"
echo "  (partial reply is still saved to history)."
echo "Prefix a user line with '-n NUM' to override max_tokens for that turn only"
echo "  (default stays at $MAX_TOKENS)."
echo "--------------------------------------------------"

# Per-turn override of MAX_TOKENS. Empty string means "use the launch default".
# Set by parsing `-n NUM` at the start of a user line; consumed and cleared on
# the next request so the default ($MAX_TOKENS) sticks again afterwards.
turn_max_tokens=""

# pipefail so a curl/jq failure inside the $(...) capture below shows up in $?.
set -o pipefail

# Ctrl+C during the streaming pipeline cancels the current turn instead of
# killing the script. The handler just sets a flag; SIGINT also reaches the
# subshell's curl/jq/tee, which die normally and end the pipeline.
CANCELLED=0
on_cancel() { CANCELLED=1; }

while true; do
  effective_max_tokens="${turn_max_tokens:-$MAX_TOKENS}"
  turn_max_tokens=""

  BODY=$(jq -n \
    --argjson messages "$MESSAGES_JSON" \
    --argjson max_tokens "$effective_max_tokens" \
    '{
      model: "deepseek-ai/DeepSeek-R1-0528",
      messages: $messages,
      max_tokens: $max_tokens,
      stream: true,
      stream_options: {include_usage: true}
    }')

  echo -n "Assistant: "

  CANCELLED=0
  trap on_cancel INT

  # Stream SSE, decode delta.content, mirror tokens to the controlling
  # terminal via /dev/tty so the user sees them live, and capture the
  # concatenated text into ASSISTANT_RESPONSE for the message history.
  # Note: `tee /dev/stderr 2>&1` does NOT work here -- /dev/stderr resolves to
  # tee's own fd 2 which after the redirect points at the captured stdout, so
  # nothing would reach the terminal.
  ASSISTANT_RESPONSE=$(curl -sS -N --fail-with-body \
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
    ' | tee /dev/tty)
  pipeline_status=$?

  trap - INT
  echo

  if [ "$CANCELLED" -eq 1 ] && [ -z "$ASSISTANT_RESPONSE" ]; then
    # Cancelled before any token arrived -- drop the user turn entirely so the
    # next request doesn't include the abandoned prompt.
    echo "[cancelled before first token — discarding last user turn]"
    MESSAGES_JSON=$(jq '.[:-1]' <<< "$MESSAGES_JSON")
  elif [ "$CANCELLED" -eq 1 ]; then
    echo "[cancelled — saving partial reply to history]"
    MESSAGES_JSON=$(jq --arg reply "$ASSISTANT_RESPONSE" \
      '. += [{role: "assistant", content: $reply}]' <<< "$MESSAGES_JSON")
  elif [ "$pipeline_status" -ne 0 ]; then
    echo "[multiturn-chat.sh] Error: request pipeline exited with status $pipeline_status." >&2
    exit "$pipeline_status"
  else
    MESSAGES_JSON=$(jq --arg reply "$ASSISTANT_RESPONSE" \
      '. += [{role: "assistant", content: $reply}]' <<< "$MESSAGES_JSON")
  fi

  echo "--------------------------------------------------"

  if ! IFS= read -r -p "User: " NEXT_PROMPT; then
    echo
    echo "Goodbye!"
    break
  fi
  if [[ "$NEXT_PROMPT" == "exit" || "$NEXT_PROMPT" == "quit" ]]; then
    echo "Goodbye!"
    break
  fi

  # Allow per-turn `-n NUM` override at the start of the line. This is a
  # one-shot: it applies only to the very next request, then $MAX_TOKENS (the
  # launch default) takes over again. The rest of the line, if any, is the
  # actual user prompt.
  if [[ "$NEXT_PROMPT" =~ ^-n[[:space:]]+([0-9]+)([[:space:]]+(.*))?$ ]]; then
    turn_max_tokens="${BASH_REMATCH[1]}"
    NEXT_PROMPT="${BASH_REMATCH[3]:-}"
    echo "[max_tokens (this turn only) = $turn_max_tokens]"
  fi

  if [ -z "$NEXT_PROMPT" ]; then
    continue
  fi

  MESSAGES_JSON=$(jq --arg prompt "$NEXT_PROMPT" \
    '. += [{role: "user", content: $prompt}]' <<< "$MESSAGES_JSON")
done
