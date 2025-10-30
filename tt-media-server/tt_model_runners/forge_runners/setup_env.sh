#!/bin/bash
# filepath: /localdev/idjuric/tt-inference-server/tt-media-server/scripts/simple_setup.sh

echo "Reseting environment..."
unset TT_METAL_HOME
unset PYTHONPATH
unset WH_ARCH_YAML
unset ARCH_NAME

virtual_env_name="venv-worker"
# Create virtual environment with available Python version
if command -v python3.11 >/dev/null 2>&1; then
    echo "Using python3.11 for virtual environment..."
    python3.11 -m venv ${virtual_env_name}
else
    echo "Error: No suitable Python version found!"
    exit 1
fi

# Activate virtual environment
source ${virtual_env_name}/bin/activate

# Install requirements
pip install --upgrade pip

# Install root requirements if exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Install forge requirements if exists
if [ -f "tt_model_runners/forge_runners/requirements.txt" ]; then
    pip install -r tt_model_runners/forge_runners/requirements.txt
fi

echo "Setup complete in virtual environment ${virtual_env_name}."
echo "To activate, run: source ${virtual_env_name}/bin/activate"