#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Setup script for running tt-inference-server locally (without Docker)
# This script sets up the required environment and dependencies

set -e

echo "Setting up local server environment for tt-inference-server..."

# 1. Set up tt-metal environment
echo "Setting up tt-metal environment..."
cd /home/tt-metal-apv
export PYTHONPATH=""
source env_vars_setup.sh

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source /opt/venv/bin/activate

# 3. Install required Python dependencies
echo "Installing Python dependencies..."
pip install pyjwt==2.7.0 requests==2.32.3 datasets==3.1.0 openai==1.53.1

# 4. Check if vLLM is available
echo "Checking for vLLM installation..."
if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vLLM is not installed locally."
    echo ""
    echo "To run the local server, you need to install vLLM and tt-metal locally."
    echo "This requires building from source and is complex."
    echo ""
    echo "RECOMMENDED: Use Docker instead:"
    echo "  python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow server --docker-server"
    echo ""
    echo "If you want to proceed with local installation, you need to:"
    echo "1. Install vLLM from source with tt-metal support"
    echo "2. Install tt-metal from source"
    echo "3. Set up all required environment variables"
    echo ""
    echo "For more details, see: https://github.com/tenstorrent/vllm/tree/dev/tt_metal"
    exit 1
fi

# 5. Check if tt-metal is available
echo "Checking for tt-metal installation..."
if ! python -c "import tt_metal" 2>/dev/null; then
    echo "ERROR: tt-metal is not installed locally."
    echo "Please install tt-metal from source first."
    exit 1
fi

echo "✅ Environment setup complete!"
echo ""
echo "You can now run the local server with:"
echo "  python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow server --local-server"

