#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
set -o pipefail

HOST="localhost"
PORT=8000
MODEL="deepseek-ai/DeepSeek-R1-0528"
MAX_TOKENS=128
API_KEY="${OPENAI_API_KEY:-your-secret-key}"
PROMPT=""

usage() {
  cat >&2 <<EOF
Usage: $0 [--dynamo] [--deepseek|--kimi|--model ID] [-p PORT] [-n MAX_TOKENS] "initial prompt"
  --dynamo        target the dynamo-frontend container (dynamo-frontend:8000; run
                  from a container on dynamo-net). Default target: localhost:$PORT.
  --deepseek      model deepseek-ai/DeepSeek-R1-0528 (default)
  --kimi          model moonshotai/Kimi-K2.6
  --model ID      explicit model id (must match GET /v1/models)
  -p PORT         server port (default: $PORT)
  -n MAX_TOKENS   max completion tokens per turn (default: $MAX_TOKENS)
Flags may appear before or after the prompt.
EOF
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dynamo)   HOST="dynamo-frontend"; PORT=8000; shift ;;
    --deepseek) MODEL="deepseek-ai/DeepSeek-R1-0528"; shift ;;
    --kimi)     MODEL="moonshotai/Kimi-K2.6"; shift ;;
    --model)    MODEL="$2"; shift 2 ;;
    -p)         PORT="$2"; shift 2 ;;
    -n)         MAX_TOKENS="$2"; shift 2 ;;
    -h|--help)  usage ;;
    --)         shift; PROMPT="${PROMPT:+$PROMPT }$*"; break ;;
    -*)         echo "Unknown flag: $1" >&2; usage ;;
    *)          PROMPT="${PROMPT:+$PROMPT }$1"; shift ;;
  esac
done
[ -n "$PROMPT" ] || usage

URL="http://${HOST}:${PORT}/v1/chat/completions"

# Stateful chat history: initial user prompt, then each turn appends the
# assistant reply and the next user line.
MESSAGES_JSON=$(jq -n --arg p "$PROMPT" '[{role:"user", content:$p}]')

echo "Target: $URL  (model: $MODEL)"
echo "Chat session — type 'exit'/'quit' to end; Ctrl+C cancels the current reply."
echo "Prefix a line with '-n NUM' to override max_tokens for that turn only."
echo "--------------------------------------------------"

turn_max_tokens=""   # one-shot per-turn -n override; cleared after each request
CANCELLED=0
on_cancel() { CANCELLED=1; }

while true; do
  effective_max_tokens="${turn_max_tokens:-$MAX_TOKENS}"; turn_max_tokens=""

  BODY=$(jq -n --arg model "$MODEL" --argjson messages "$MESSAGES_JSON" \
    --argjson max_tokens "$effective_max_tokens" \
    '{model:$model, messages:$messages, max_tokens:$max_tokens,
      stream:true, stream_options:{include_usage:true}}')

  echo -n "Assistant: "
  CANCELLED=0; trap on_cancel INT

  # Stream SSE, decode delta.content, mirror it live to the terminal via
  # /dev/tty, and capture the concatenated text for the message history.
  # (`tee /dev/stderr` wouldn't reach the terminal — fd 2 points at the capture.)
  ASSISTANT_RESPONSE=$(curl -sS -N --fail-with-body -X POST "$URL" \
      -H "Authorization: Bearer ${API_KEY}" \
      -H "Content-Type: application/json" \
      -d "$BODY" \
    | jq --raw-input --unbuffered -j '
        select(startswith("data: ")) | sub("^data: "; "") | select(. != "[DONE]")
        | (fromjson? // {}) | .choices[0].delta.content // ""' \
    | tee /dev/tty)
  pipeline_status=$?
  trap - INT
  echo

  if [ "$CANCELLED" -eq 1 ] && [ -z "$ASSISTANT_RESPONSE" ]; then
    echo "[cancelled before first token — discarding this turn]"
    MESSAGES_JSON=$(jq '.[:-1]' <<< "$MESSAGES_JSON")
  elif [ "$CANCELLED" -eq 0 ] && [ "$pipeline_status" -ne 0 ]; then
    echo "[multiturn-chat.sh] request pipeline exited with status $pipeline_status" >&2
    exit "$pipeline_status"
  else
    # Normal completion, or cancelled with a partial reply: save what we got.
    MESSAGES_JSON=$(jq --arg r "$ASSISTANT_RESPONSE" \
      '. += [{role:"assistant", content:$r}]' <<< "$MESSAGES_JSON")
  fi

  echo "--------------------------------------------------"
  IFS= read -r -p "User: " NEXT_PROMPT || { echo; echo "Goodbye!"; break; }
  [[ "$NEXT_PROMPT" == "exit" || "$NEXT_PROMPT" == "quit" ]] && { echo "Goodbye!"; break; }

  # Optional one-shot `-n NUM` at the start of a line overrides max_tokens for
  # the next request only; the rest of the line is the prompt.
  if [[ "$NEXT_PROMPT" =~ ^-n[[:space:]]+([0-9]+)([[:space:]]+(.*))?$ ]]; then
    turn_max_tokens="${BASH_REMATCH[1]}"; NEXT_PROMPT="${BASH_REMATCH[3]:-}"
    echo "[max_tokens this turn = $turn_max_tokens]"
  fi
  [ -n "$NEXT_PROMPT" ] || continue

  MESSAGES_JSON=$(jq --arg p "$NEXT_PROMPT" \
    '. += [{role:"user", content:$p}]' <<< "$MESSAGES_JSON")
done
