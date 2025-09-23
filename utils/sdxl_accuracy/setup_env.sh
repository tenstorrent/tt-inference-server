#!/bin/bash

set -e  # Exit on error

# Allow overriding Python command
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD="python3"
fi

# Verify Python command exists
if ! command -v $PYTHON_CMD &>/dev/null; then
    echo "Python command not found: $PYTHON_CMD"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set Python environment directory
if [ -z "$PYTHON_ENV_DIR" ]; then
    PYTHON_ENV_DIR="$SCRIPT_DIR/.sdxl_accuracy_env"
fi
echo "Creating virtual env in: $PYTHON_ENV_DIR"

# Create and activate virtual environment
$PYTHON_CMD -m venv $PYTHON_ENV_DIR
source $PYTHON_ENV_DIR/bin/activate

# Install requirements
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "✅ Setup complete!"
echo "To activate: source $PYTHON_ENV_DIR/bin/activate"