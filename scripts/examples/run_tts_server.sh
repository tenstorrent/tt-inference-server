#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Script to run SpeechT5 TTS server on Tenstorrent N150 hardware
# 
# Usage:
#   ./run_tts_server.sh [port]
#
# Examples:
#   ./run_tts_server.sh           # Start on default port 8000
#   ./run_tts_server.sh 8080      # Start on port 8080

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default port
PORT=${1:-8000}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  SpeechT5 TTS Server Setup (N150)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if we're in the correct directory
if [ ! -f "main.py" ]; then
    echo -e "${RED}Error: main.py not found${NC}"
    echo "Please run this script from the tt-media-server directory"
    echo ""
    echo "Example:"
    echo "  cd /path/to/tt-inference-server/tt-media-server"
    echo "  bash ../scripts/examples/run_tts_server.sh"
    exit 1
fi

# Check if tt-metal is set up
if [ -z "$TT_METAL_HOME" ]; then
    echo -e "${YELLOW}Warning: TT_METAL_HOME not set${NC}"
    echo "Attempting to auto-detect..."
    
    # Try to find tt-metal in parent directories
    if [ -d "../../tt_metal" ]; then
        export TT_METAL_HOME=$(cd ../.. && pwd)
        echo -e "${GREEN}Found tt-metal at: $TT_METAL_HOME${NC}"
    else
        echo -e "${RED}Error: Could not find tt-metal installation${NC}"
        echo "Please set TT_METAL_HOME environment variable:"
        echo "  export TT_METAL_HOME=/path/to/tt-metal"
        exit 1
    fi
fi

# Check if Python virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: No Python virtual environment detected${NC}"
    echo "Attempting to activate tt-metal venv..."
    
    if [ -f "$TT_METAL_HOME/python_env/bin/activate" ]; then
        source "$TT_METAL_HOME/python_env/bin/activate"
        echo -e "${GREEN}Activated Python virtual environment${NC}"
    else
        echo -e "${RED}Error: Could not find Python virtual environment${NC}"
        echo "Please activate the tt-metal Python environment:"
        echo "  source $TT_METAL_HOME/python_env/bin/activate"
        exit 1
    fi
fi

# Check for Tenstorrent device
echo "Checking for Tenstorrent hardware..."
if ! command -v tt-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: tt-smi not found${NC}"
    echo "Cannot verify device status"
else
    if ! tt-smi &> /dev/null; then
        echo -e "${RED}Error: No Tenstorrent devices detected${NC}"
        echo "Please ensure N150 hardware is properly installed"
        exit 1
    fi
    echo -e "${GREEN}✓ Tenstorrent device detected${NC}"
fi

# Set up environment variables for N150 TTS
echo ""
echo "Setting up N150 TTS configuration..."

export ARCH_NAME=${ARCH_NAME:-wormhole_b0}
export PYTHONPATH=$TT_METAL_HOME:$(pwd)

# N150-specific TTS configuration
export MODEL_RUNNER=tt-speecht5-tts
export MODEL_SERVICE=audio
export DEVICE_IDS="(0)"
export IS_GALAXY=False

echo -e "${GREEN}Environment variables set:${NC}"
echo "  ARCH_NAME=$ARCH_NAME"
echo "  TT_METAL_HOME=$TT_METAL_HOME"
echo "  PYTHONPATH=$PYTHONPATH"
echo "  MODEL_RUNNER=$MODEL_RUNNER"
echo "  MODEL_SERVICE=$MODEL_SERVICE"
echo "  DEVICE_IDS=$DEVICE_IDS"
echo "  IS_GALAXY=$IS_GALAXY"
echo ""

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}Warning: Port $PORT is already in use${NC}"
    echo "Please choose a different port or stop the existing service"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Starting TTS Server${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Server will be available at: http://localhost:$PORT"
echo ""
echo -e "${YELLOW}Note: First startup will take 30-60 seconds${NC}"
echo "      Wait for: 'All workers warmed up and ready'"
echo ""
echo "To test the server, open a new terminal and run:"
echo ""
echo -e "${GREEN}curl -X POST \"http://localhost:$PORT/tts/tts\" \\${NC}"
echo -e "${GREEN}  -H \"Authorization: Bearer your-secret-key\" \\${NC}"
echo -e "${GREEN}  -H \"Content-Type: application/json\" \\${NC}"
echo -e "${GREEN}  -d '{\"text\": \"Hello world\", \"stream\": false}' \\${NC}"
echo -e "${GREEN}  -o output.wav${NC}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo -e "${GREEN}========================================${NC}"
echo ""

# Start the server
uvicorn main:app --lifespan on --port $PORT --host 0.0.0.0




