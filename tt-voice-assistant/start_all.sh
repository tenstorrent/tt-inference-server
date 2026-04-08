#!/bin/bash
# Start all TT Voice Assistant servers
# Run this inside the Docker container

set -e

# Environment setup
cd /home/container_app_user/tt-metal
source python_env/bin/activate
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH"
export TT_METAL_HOME=/home/container_app_user/tt-metal
export HF_HOME=/home/container_app_user/.cache/huggingface

# P150 mesh descriptor (single chip mode)
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto

VOICE_APP_DIR="/home/container_app_user/voice-assistant"

echo "=========================================="
echo "Starting TT Voice Assistant Pipeline"
echo "=========================================="

# Kill any existing servers
echo "Cleaning up old processes..."
pkill -f "face_auth_server.py" || true
pkill -f "whisper_server.py" || true
pkill -f "speecht5_server.py" || true
pkill -f "llama_server.py" || true
sleep 2

# Start Face Auth Server (Device 0)
echo ""
echo "[1/4] Starting Face Auth Server (Device 0)..."
TT_VISIBLE_DEVICES='0' python $VOICE_APP_DIR/servers/face_auth_server.py &
sleep 5

# Start Whisper Server (Device 2)
echo ""
echo "[2/4] Starting Whisper Server (Device 2)..."
TT_VISIBLE_DEVICES='2' python $VOICE_APP_DIR/servers/whisper_server.py &
sleep 5

# Start TTS Server (CPU - SpeechT5)
echo ""
echo "[3/4] Starting TTS Server (SpeechT5 on CPU)..."
python $VOICE_APP_DIR/servers/speecht5_server.py &
sleep 3

# Start Llama Server (Device 1)
echo ""
echo "[4/4] Starting Llama Server (Device 1)..."
TT_VISIBLE_DEVICES='1' python $VOICE_APP_DIR/servers/llama_server.py &
sleep 5

echo ""
echo "=========================================="
echo "All servers starting in background"
echo "=========================================="
echo ""
echo "Wait for all servers to warm up (~2-3 min)..."
echo "Then start the main app:"
echo ""
echo "  cd $VOICE_APP_DIR && python main.py"
echo ""
echo "Access UI at: http://localhost:8000"
echo ""

# Wait for all background processes
wait
