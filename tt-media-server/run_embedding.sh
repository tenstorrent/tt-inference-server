#!/bin/bash

# Script to send 4 embedding requests with different input lengths
# Update your-secret-key with the real API key before running.

URL="http://localhost:8011/v1/embeddings"
AUTH="Authorization: Bearer your-secret-key"
CONTENT="Content-Type: application/json"

# Function to send a POST request
send_request() {
  local INPUT_TEXT="$1"
  local LABEL="$2"

  echo "=== Running request for $LABEL tokens ==="
  curl -s -X POST "$URL" \
    -H "$CONTENT" \
    -H "$AUTH" \
    -d "{\"input\": \"$INPUT_TEXT\", \"dimensions\": 10}" \
    | jq '.'
  echo
}

# Define repeated texts
HELLO_128="$(printf 'hello %.0s' {1..120})"
HELLO_256="$(printf 'hello %.0s' {1..250})"
HELLO_512="$(printf 'hello %.0s' {1..510})"
HELLO_1024="$(printf 'hello %.0s' {1..1020})"
HELLO_2048="$(printf 'hello %.0s' {1..2040})"
HELLO_4096="$(printf 'hello %.0s' {1..4090})"

# Run all requests
send_request "$HELLO_128" "128"
send_request "$HELLO_256" "256"
send_request "$HELLO_512" "512"
send_request "$HELLO_1024" "1024"
send_request "$HELLO_2048" "2048"
send_request "$HELLO_4096" "4096"

