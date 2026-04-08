#!/bin/bash
# TT Voice Assistant - Single command to start everything
# Usage: ./start.sh

set -e

CONTAINER="tt-voice-test"
IMAGE="tt-voice-assistant:push-button"
LOCAL_APP="/home/ttuser/teja/tt-voice-assistant/app"

echo "=========================================="
echo "  TT Voice Assistant - Starting Pipeline"
echo "=========================================="

# 1. Check if container exists, create if not
echo ""
echo "[1/6] Checking container..."
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "    Container not found, creating..."
    docker run -d --name $CONTAINER \
        --device /dev/tenstorrent:/dev/tenstorrent \
        -v /dev/hugepages-1G:/dev/hugepages-1G \
        -v /home/ttuser/teja/tt-voice-assistant:/home/ttuser/teja/tt-voice-assistant \
        -p 8080:8080 \
        --entrypoint /bin/bash \
        $IMAGE \
        -c "sleep infinity"
    echo "    Container created"
    sleep 5
elif ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "    Container exists but stopped, starting..."
    docker start $CONTAINER
    sleep 3
else
    echo "    Container already running"
fi

# 2. Stop any existing processes inside container
echo "[2/6] Stopping existing processes..."
docker exec $CONTAINER pkill -f "face_auth_server.py" 2>/dev/null || true
docker exec $CONTAINER pkill -f "whisper_server.py" 2>/dev/null || true
docker exec $CONTAINER pkill -f "speecht5_server.py" 2>/dev/null || true
docker exec $CONTAINER pkill -f "tts_server.py" 2>/dev/null || true
docker exec $CONTAINER pkill -f "main.py" 2>/dev/null || true
sleep 2

# 3. Install dependencies (onnx for SFace, ffmpeg for audio conversion)
echo "[3/7] Installing dependencies..."
docker exec $CONTAINER bash -c 'apt-get update -qq && apt-get install -y -qq ffmpeg 2>/dev/null || true'
docker exec $CONTAINER bash -c 'pip install onnx -q 2>/dev/null || true'

# 4. Copy latest code to container
echo "[4/7] Copying latest code to container..."
docker cp $LOCAL_APP/main_fixed.py $CONTAINER:/home/container_app_user/voice-assistant/main.py
docker cp $LOCAL_APP/services/. $CONTAINER:/home/container_app_user/voice-assistant/services/
docker cp $LOCAL_APP/templates/. $CONTAINER:/home/container_app_user/voice-assistant/templates/
docker cp $LOCAL_APP/registered_faces/. $CONTAINER:/home/container_app_user/voice-assistant/registered_faces/ 2>/dev/null || true

# 5. Start Face Auth Server (Device 0)
echo "[5/7] Starting Face Auth Server (Device 0)..."
docker exec -d $CONTAINER bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH"
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=0
cd /home/container_app_user/tt-metal
python /home/ttuser/teja/tt-voice-assistant/app/servers/face_auth_server.py > /tmp/face_auth_server.log 2>&1
'

# 6. Start Whisper Server (Device 2)
echo "[6/7] Starting Whisper Server (Device 2)..."
docker exec -d $CONTAINER bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH"
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=2
cd /home/container_app_user/tt-metal
python /home/ttuser/teja/tt-voice-assistant/app/servers/whisper_server.py > /tmp/whisper_server.log 2>&1
'

# Start TTS Server (SpeechT5 TTNN on Device 3)
echo "       Starting TTS Server (SpeechT5 TTNN - Device 3)..."
docker exec -d $CONTAINER bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH"
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
cd /home/container_app_user/tt-metal
python -u /home/ttuser/teja/tt-voice-assistant/app/servers/speecht5_ttnn_server.py > /tmp/tts_server.log 2>&1
'

# Wait for servers to start
echo "       Waiting for servers to initialize..."
sleep 10

# 7. Start Main App (Llama on Device 1)
echo "[7/7] Starting Main App (Llama on Device 1)..."
docker exec -d $CONTAINER bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH"
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=1
cd /home/container_app_user/voice-assistant
python main.py > /tmp/main_app.log 2>&1
'

echo ""
echo "=========================================="
echo "  All services starting!"
echo "=========================================="
echo ""
echo "Wait ~60-90 seconds for models to load, then:"
echo "  - Face Auth: http://localhost:8080/"
echo "  - Chat UI:   http://localhost:8080/chat"
echo ""
echo "Check logs:"
echo "  docker exec $CONTAINER tail -30 /tmp/main_app.log"
echo "  docker exec $CONTAINER tail -30 /tmp/face_auth_server.log"
echo "  docker exec $CONTAINER tail -30 /tmp/whisper_server.log"
echo "  docker exec $CONTAINER tail -30 /tmp/tts_server.log"
echo ""
