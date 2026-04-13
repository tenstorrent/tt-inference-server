#!/bin/bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONTAINER_NAME="Llama-3.1-8B-Instruct"

# Stop any existing container with this name
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Get the docker run command from run.py without executing it
RAW_OUTPUT=$(cd "$REPO" && "$REPO/venv/bin/python" run.py \
  --model Llama-3.1-8B-Instruct \
  --device p300 \
  --workflow server \
  --docker-server \
  --dev-mode \
  --device-id 0,1 \
  --service-port 7000 \
  --host-hf-cache ~/.cache/huggingface/ \
  --skip-system-sw-validation \
  --print-docker-cmd 2>&1)

# Extract docker command (skip logger output + "Docker run command:" header),
# replace auto-generated UUID name with fixed name, inject -d for detached mode
DOCKER_CMD=$(echo "$RAW_OUTPUT" \
  | sed -n '/^docker run/,$p' \
  | sed "s/--name [^ ]*/--name $CONTAINER_NAME/" \
  | sed 's/^docker run/docker run -d/')

if [ -z "$DOCKER_CMD" ]; then
  echo "ERROR: failed to extract docker command from run.py output" >&2
  echo "Raw output:" >&2
  echo "$RAW_OUTPUT" >&2
  exit 1
fi

eval "$DOCKER_CMD"

docker network connect tt_studio_network "$CONTAINER_NAME"
echo "$CONTAINER_NAME started on host port 7000"
docker logs -f "$CONTAINER_NAME"
