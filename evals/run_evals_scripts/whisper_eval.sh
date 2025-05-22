#!/bin/bash

# Exit on error, use of unset vars, and failure in piped commands
set -euo pipefail

# Log stdout and stderr
LOG_FILE="whisper_debug_librispeech_full.txt"
exec > >(tee "$LOG_FILE") 2>&1

# Resolve absolute path to parent directory
PARENT_DIR="$(cd .. && pwd)"
echo "Using parent directory: $PARENT_DIR"

# Ensure container_app_user owns the directory where we'll work
echo "Fixing ownership of $PARENT_DIR..."
chown -R container_app_user:container_app_user "$PARENT_DIR"

# Forward all environment variables to the user shell by sourcing the temp file
echo "Switching to container_app_user and running commands..."
su - container_app_user -c "bash -i -c '
  set -euo pipefail

  echo Changing directory to parent...
  cd \"$PARENT_DIR\"

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
  lmms-eval --model=whisper_tt --model_args pretrained=distil-whisper/distil-large-v3 --task librispeech --log_samples --output_path ./logs/
'"

