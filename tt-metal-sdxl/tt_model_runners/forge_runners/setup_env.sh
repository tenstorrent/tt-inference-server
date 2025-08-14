#!/bin/bash
# filepath: /localdev/idjuric/tt-inference-server/tt-metal-sdxl/scripts/simple_setup.sh

echo "Reseting environment..."
unset TT_METAL_HOME
unset PYTHONPATH
unset WH_ARCH_YAML
unset ARCH_NAME

# Create virtual environment
python3 -m venv venv-worker

# Activate virtual environment
source venv-worker/bin/activate

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

echo "Setup complete! Virtual environment is activated."