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
# API mode: "chat" -> /v1/chat/completions (default), "responses" -> /v1/responses.
MODE="chat"

usage() {
  cat >&2 <<EOF
Usage: $0 [--dynamo] [--deepseek|--kimi|--model ID] [--responses] [-p PORT] [-n MAX_TOKENS] "initial prompt"
  --dynamo        target the dynamo-frontend container (dynamo-frontend:8000; run
                  from a container on dynamo-net). Default target: localhost:$PORT.
  --deepseek      model deepseek-ai/DeepSeek-R1-0528 (default)
  --kimi          model moonshotai/Kimi-K2.6
  --model ID      explicit model id (must match GET /v1/models)
  --responses     use the /v1/responses API (default: /v1/chat/completions).
                  Multi-turn state is carried server-side via previous_response_id
                  (the id of the prior turn) instead of a local message history.
  -p PORT         server port (default: $PORT)
  -n MAX_TOKENS   max completion tokens per turn (default: $MAX_TOKENS)
Flags may appear before or after the prompt.
EOF
  exit 1
}

# Manual arg loop so flags work in any position.
while [ $# -gt 0 ]; do
  case "$1" in
    --dynamo)    HOST="dynamo-frontend"; PORT=8000; shift ;;
    --deepseek)  MODEL="deepseek-ai/DeepSeek-R1-0528"; shift ;;
    --kimi)      MODEL="moonshotai/Kimi-K2.6"; shift ;;
    --model)     MODEL="$2"; shift 2 ;;
    --responses) MODE="responses"; shift ;;
    -p)          PORT="$2"; shift 2 ;;
    -n)          MAX_TOKENS="$2"; shift 2 ;;
    -h|--help)   usage ;;
    --)          shift; PROMPT="${PROMPT:+$PROMPT }$*"; break ;;
    -*)          echo "Unknown flag: $1" >&2; usage ;;
    *)           PROMPT="${PROMPT:+$PROMPT }$1"; shift ;;
  esac
done
[ -n "$PROMPT" ] || usage

if [ "$MODE" = "responses" ]; then
  URL="http://${HOST}:${PORT}/v1/responses"
else
  URL="http://${HOST}:${PORT}/v1/chat/completions"
fi

# State per mode:
#   chat:      full message history; we resend it every turn.
#   responses: only the current user input is sent; prior turns are referenced
#              server-side via previous_response_id, so we just track that id.
MESSAGES_JSON=$(jq -n --arg p "$PROMPT" '[{role:"user", content:$p}]')
CURRENT_INPUT="$PROMPT"
PREV_RESPONSE_ID=""

# Streaming text decoder, one per API shape. Reads raw SSE lines on stdin and
# emits just the assistant text deltas.
if [ "$MODE" = "responses" ]; then
  TEXT_FILTER='
    select(startswith("data: "))
    | sub("^data: "; "")
    | select(. != "[DONE]")
    | (fromjson? // {})
    | select(.type == "response.output_text.delta")
    | .delta // ""'
else
  TEXT_FILTER='
    select(startswith("data: "))
    | sub("^data: "; "")
    | select(. != "[DONE]")
    | (fromjson? // {})
    | .choices[0].delta.content // ""'
fi

echo "Target: $URL  (model: $MODEL)"
echo "Chat session (${MODE} API) — type 'exit'/'quit' to end; Ctrl+C cancels the current reply."
echo "Prefix a line with '-n NUM' to override max_tokens for that turn only."
echo "--------------------------------------------------"

turn_max_tokens=""   # one-shot per-turn -n override; cleared after each request
CANCELLED=0
on_cancel() { CANCELLED=1; }

while true; do
  effective_max_tokens="${turn_max_tokens:-$MAX_TOKENS}"; turn_max_tokens=""

  if [ "$MODE" = "responses" ]; then
    # Only the current input is sent; previous_response_id chains prior turns
    # (omitted on the first turn). store:true is required for the server to
    # retain this response so the next turn can reference it.
    BODY=$(jq -n \
      --arg model "$MODEL" \
      --arg input "$CURRENT_INPUT" \
      --argjson max_tokens "$effective_max_tokens" \
      --arg prev "$PREV_RESPONSE_ID" \
      '{
        model: $model,
        input: $input,
        max_output_tokens: $max_tokens,
        store: true,
        stream: true
      }
      + (if $prev == "" then {} else {previous_response_id: $prev} end)')
  else
    BODY=$(jq -n --arg model "$MODEL" --argjson messages "$MESSAGES_JSON" \
      --argjson max_tokens "$effective_max_tokens" \
      '{model:$model, messages:$messages, max_tokens:$max_tokens,
        stream:true, stream_options:{include_usage:true}}')
  fi

  echo -n "Assistant: "
  CANCELLED=0; trap on_cancel INT

  # Capture the raw SSE stream to a temp file (so we can recover the response
  # id afterward in responses mode), decode delta text, mirror it live to the
  # terminal via /dev/tty, and capture the concatenated text for the history.
  # (`tee /dev/stderr` wouldn't reach the terminal — fd 2 points at the capture.)
  RAW_SSE=$(mktemp)
  ASSISTANT_RESPONSE=$(curl -sS -N --fail-with-body -X POST "$URL" \
      -H "Authorization: Bearer ${API_KEY}" \
      -H "Content-Type: application/json" \
      -d "$BODY" \
    | tee "$RAW_SSE" \
    | jq --raw-input --unbuffered -j "$TEXT_FILTER" \
    | tee /dev/tty)
  pipeline_status=$?
  trap - INT
  echo

  # In responses mode, pull the id of the just-completed response out of the
  # raw stream. Every event carries .response.id; the last one belongs to the
  # terminal (completed) event, which is the id we chain from next turn.
  NEW_RESPONSE_ID=""
  if [ "$MODE" = "responses" ]; then
    NEW_RESPONSE_ID=$(jq -r --raw-input '
      select(startswith("data: "))
      | sub("^data: "; "")
      | select(. != "[DONE]")
      | (fromjson? // {})
      | .response.id // empty' "$RAW_SSE" | tail -n1)
  fi
  rm -f "$RAW_SSE"

  if [ "$CANCELLED" -eq 1 ] && [ -z "$ASSISTANT_RESPONSE" ]; then
    # Cancelled before any token arrived.
    if [ "$MODE" = "responses" ]; then
      # No completed response was stored, so there is nothing new to chain
      # from -- keep the previous id and move on.
      echo "[cancelled before first token]"
    else
      # Drop the user turn entirely so the next request doesn't include the
      # abandoned prompt.
      echo "[cancelled before first token — discarding this turn]"
      MESSAGES_JSON=$(jq '.[:-1]' <<< "$MESSAGES_JSON")
    fi
  elif [ "$CANCELLED" -eq 1 ]; then
    if [ "$MODE" = "responses" ]; then
      # A cancelled turn has no completed (stored) response id to chain from,
      # so keep the previous id; the partial reply lives only on screen.
      echo "[cancelled — partial reply not chained to next turn]"
    else
      echo "[cancelled — saving partial reply to history]"
      MESSAGES_JSON=$(jq --arg r "$ASSISTANT_RESPONSE" \
        '. += [{role:"assistant", content:$r}]' <<< "$MESSAGES_JSON")
    fi
  elif [ "$pipeline_status" -ne 0 ]; then
    echo "[multiturn-chat.sh] request pipeline exited with status $pipeline_status" >&2
    exit "$pipeline_status"
  else
    if [ "$MODE" = "responses" ]; then
      if [ -n "$NEW_RESPONSE_ID" ]; then
        PREV_RESPONSE_ID="$NEW_RESPONSE_ID"
      else
        echo "[multiturn-chat.sh] Warning: no response id found in stream;" \
             "next turn will start a fresh conversation." >&2
        PREV_RESPONSE_ID=""
      fi
    else
      MESSAGES_JSON=$(jq --arg r "$ASSISTANT_RESPONSE" \
        '. += [{role:"assistant", content:$r}]' <<< "$MESSAGES_JSON")
    fi
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

  if [ "$MODE" = "responses" ]; then
    CURRENT_INPUT="$NEXT_PROMPT"
  else
    MESSAGES_JSON=$(jq --arg p "$NEXT_PROMPT" \
      '. += [{role:"user", content:$p}]' <<< "$MESSAGES_JSON")
  fi
done
