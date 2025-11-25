send_request() {
  local INPUT_TEXT="$1"
  local LABEL="$2"

  echo "=== Running request for $LABEL tokens ==="
  # send request, discard response body but keep timing and show errors
  time curl -sS -o /dev/null -X POST "http://localhost:8010/v1/embeddings" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer your-secret-key" \
    -d "{\"input\": \"$INPUT_TEXT\"}"
}

# inputs
HELLO_128="$(printf 'hello %.0s' {1..120})"
HELLO_256="$(printf 'hello %.0s' {1..250})"
HELLO_512="$(printf 'hello %.0s' {1..510})"
HELLO_1024="$(printf 'hello %.0s' {1..1020})"
HELLO_2048="$(printf 'hello %.0s' {1..2040})"
HELLO_4096="$(printf 'hello %.0s' {1..4090})"
HELLO_8192="$(printf 'hello %.0s' {1..8189})"
HELLO_16384="$(printf 'hello %.0s' {1..16380})"

# sending requests
send_request "$HELLO_128" "128"
send_request "$HELLO_256" "256"
send_request "$HELLO_512" "512"
send_request "$HELLO_1024" "1024"
send_request "$HELLO_2048" "2048"
send_request "$HELLO_4096" "4096"
send_request "$HELLO_8192" "8192"
send_request "$HELLO_16384" "16384"