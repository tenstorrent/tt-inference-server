#!/bin/bash

# Exit on error, use of unset vars, and failure in piped commands
set -euo pipefail

# Log stdout and stderr
LOG_FILE="whisper_debug_librispeech_full.txt"
exec > >(tee "$LOG_FILE") 2>&1

# Get current working directory
CURRENT_DIR="$(pwd)"
echo "Using current directory: $CURRENT_DIR"

# Ensure container_app_user owns the current directory
echo "Fixing ownership of $CURRENT_DIR..."
chown -R container_app_user:container_app_user "$CURRENT_DIR"

# Forward all environment variables to the user shell by sourcing the temp file
echo "Switching to container_app_user and running commands..."
su - container_app_user -c "bash -i -c '
  set -euo pipefail

  echo Checking for lmms-eval directory in current location...
  if [ ! -d \"lmms-eval/.git\" ]; then
    echo \"Cloning lmms-eval repository...\"
    git clone https://github.com/bgoelTT/lmms-eval.git --branch ben/whisper-tt
  else
    echo \"Repository already exists. Skipping clone.\"
  fi

  echo Changing directory to lmms-eval...
  cd lmms-eval

  echo Installing Python package...
  pip install -e .

  echo Installing audio extras...
  pip install -e .[audio]

  echo Running lmms-eval...
  export WH_ARCH_YAML="${WH_ARCH_YAML:-}"
  export TT_METAL_HOME=$TT_METAL_HOME
  export HF_TOKEN=$HF_TOKEN
  export PYTHONPATH=$PYTHONPATH
  export ARCH_NAME=$ARCH_NAME
'"
