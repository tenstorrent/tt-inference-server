#!/bin/bash
# filepath: /localdev/idjuric/tt-inference-server/tt-media-server/scripts/simple_setup.sh

echo "Reseting environment..."
unset TT_METAL_HOME
unset PYTHONPATH
unset WH_ARCH_YAML
unset ARCH_NAME

# Create virtual environment with available Python version
if command -v python3.11 >/dev/null 2>&1; then
    echo "Using python3.11 for virtual environment..."
    python3.11 -m venv venv-worker
elif command -v python3 >/dev/null 2>&1; then
    echo "Using python3 for virtual environment..."
    python3 -m venv venv-worker
else
    echo "Error: No suitable Python version found!"
    exit 1
fi

# Activate virtual environment
source venv-worker/bin/activate

# Install requirements
pip3 install --upgrade pip

# Install root requirements if exists
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
fi

# Install forge requirements if exists
if [ -f "tt_model_runners/forge_runners/requirements.txt" ]; then
    pip3 install -r tt_model_runners/forge_runners/requirements.txt
fi

echo "Setup complete! Virtual environment is activated."