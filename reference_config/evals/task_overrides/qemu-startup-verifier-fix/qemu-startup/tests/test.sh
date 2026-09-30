#!/bin/bash

# The task image already contains wget, expect, and ca-certificates. Installing
# curl/expect here used stale Debian bullseye package indexes and prevented the
# actual pytest verifier from running, even when the agent completed the task.
set -o pipefail

installer=/tmp/uv-installer.sh
if ! wget -q https://astral.sh/uv/0.9.5/install.sh -O "$installer"; then
  echo "Failed to download the uv installer" >&2
  exit 1
fi
if ! sh "$installer"; then
  echo "Failed to install uv" >&2
  exit 1
fi
export PATH="$HOME/.local/bin:$PATH"

# Check if we're in a valid working directory
if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set. Please set a WORKDIR in your Dockerfile before running this script."
    exit 1
fi

if uvx \
  -p 3.13 \
  -w pytest==8.4.1 \
  -w pytest-json-ctrf==0.3.5 \
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
