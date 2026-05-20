#!/bin/bash
# TT-HOME Voice Assistant - Wormhole Galaxy (SpeechT5 version)
# Usage: ./start.sh
#
# ===== Architecture =====
# Hardware: Wormhole Galaxy (32 N150 chips)
# Docker: push-button image (old tt-metal with working SpeechT5 traces)
#
# Device allocation:
#   Llama 3.1-8B:       N150 (1 chip, device 0) - fabric disabled
#   Whisper distil-v3:  N150 (1 chip, device 2) - MeshShape 1x1, use_trace=True
#   SpeechT5 TTS HOST: N150 (1 chip, device 4) - traced (~150-195 tok/s)
#   SpeechT5 TTS GUEST: N150 (1 chip, device 6) - traced, podcast male voice
#
# TTS Backend: SpeechT5 (TTNN accelerated with traces)
# ==============================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="tt-wh-glxy-v2"
IMAGE="ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:push-button"
LOCAL_APP="$SCRIPT_DIR"

N150_MESH_DESC="/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.textproto"

LLAMA_DEVICES="0"
WHISPER_DEVICES="2"
TTS_HOST_DEVICES="4"
TTS_GUEST_DEVICES="6"

echo "=========================================="
echo "  TT-HOME Voice Assistant (Wormhole Galaxy)"
echo "  TTS: SpeechT5 (traced) | LLM: Llama 3.1-8B"
echo "=========================================="

# 1. Check if container exists, create if not
echo ""
echo "[1/5] Checking container..."
if ! sg docker -c "docker ps -a --format '{{.Names}}'" | grep -q "^${CONTAINER}$"; then
    echo "    Container not found, creating..."
    sg docker -c "docker run -d --name $CONTAINER \
        --device /dev/tenstorrent:/dev/tenstorrent \
        -v /dev/hugepages-1G:/dev/hugepages-1G \
        -v \"$SCRIPT_DIR\":\"$SCRIPT_DIR\" \
        -p 8080:8080 \
        --entrypoint /bin/bash \
        $IMAGE \
        -c 'sleep infinity'"
    echo "    Container created"
    sleep 5
elif ! sg docker -c "docker ps --format '{{.Names}}'" | grep -q "^${CONTAINER}$"; then
    echo "    Container exists but stopped, starting..."
    sg docker -c "docker start $CONTAINER"
    sleep 3
else
    echo "    Container already running"
fi

# 2. Stop any existing processes inside container
echo "[2/5] Stopping existing processes..."
sg docker -c "docker exec $CONTAINER pkill -f 'whisper_server.py'" 2>/dev/null || true
sg docker -c "docker exec $CONTAINER pkill -f 'speecht5_ttnn_server.py'" 2>/dev/null || true
sg docker -c "docker exec $CONTAINER pkill -f 'main.py'" 2>/dev/null || true
sg docker -c "docker exec $CONTAINER rm -f /tmp/tts_server.sock /tmp/tts_server_guest.sock /tmp/whisper_server.sock" 2>/dev/null || true
sleep 2

# 3. Install dependencies & copy code
echo "[3/5] Installing dependencies and copying code..."
sg docker -c "docker exec $CONTAINER bash -c 'source /home/container_app_user/tt-metal/python_env/bin/activate && pip install onnx PyPDF2 requests beautifulsoup4 num2words uvicorn fastapi python-multipart -q 2>/dev/null || true'"
sg docker -c "docker exec $CONTAINER bash -c 'apt-get update -qq && apt-get install -y -qq ffmpeg 2>/dev/null || true'"

sg docker -c "docker cp $LOCAL_APP/main.py $CONTAINER:/home/container_app_user/voice-assistant/main.py"
sg docker -c "docker cp $LOCAL_APP/services/. $CONTAINER:/home/container_app_user/voice-assistant/services/"
sg docker -c "docker cp $LOCAL_APP/templates/. $CONTAINER:/home/container_app_user/voice-assistant/templates/"
sg docker -c "docker cp $LOCAL_APP/servers/. $CONTAINER:/home/container_app_user/voice-assistant/servers/"
sg docker -c "docker cp $LOCAL_APP/static/. $CONTAINER:/home/container_app_user/voice-assistant/static/"
sg docker -c "docker exec $CONTAINER mkdir -p /home/container_app_user/voice-assistant/output"
sg docker -c "docker exec $CONTAINER mkdir -p /home/container_app_user/voice-assistant/logs"

# 4. Start Whisper + SpeechT5 TTS servers
echo "[4/5] Starting model servers..."

echo "       Whisper (Device $WHISPER_DEVICES, N150, traced)..."
sg docker -c "docker exec -d $CONTAINER bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=$N150_MESH_DESC
export TT_VISIBLE_DEVICES=$WHISPER_DEVICES
cd /home/container_app_user/tt-metal
python /home/container_app_user/voice-assistant/servers/whisper_server.py > /tmp/whisper_server.log 2>&1
'"

echo "       SpeechT5 TTS HOST (Device $TTS_HOST_DEVICES, N150, traced, speaker 7306)..."
sg docker -c "docker exec -d $CONTAINER bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=$N150_MESH_DESC
export TT_VISIBLE_DEVICES=$TTS_HOST_DEVICES
cd /home/container_app_user/tt-metal
python /home/container_app_user/voice-assistant/servers/speecht5_ttnn_server.py --speaker-id 7306 --socket /tmp/tts_server.sock > /tmp/tts_server.log 2>&1
'"

echo "       SpeechT5 TTS GUEST (Device $TTS_GUEST_DEVICES, N150, traced, speaker 1138)..."
sg docker -c "docker exec -d $CONTAINER bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=$N150_MESH_DESC
export TT_VISIBLE_DEVICES=$TTS_GUEST_DEVICES
cd /home/container_app_user/tt-metal
python /home/container_app_user/voice-assistant/servers/speecht5_ttnn_server.py --speaker-id 1138 --socket /tmp/tts_server_guest.sock > /tmp/tts_server_guest.log 2>&1
'"

echo "       Waiting for servers to initialize..."
sleep 15

# 5. Start Main App (Llama on N150)
echo "[5/5] Starting Main App (Llama on Device $LLAMA_DEVICES, N150)..."
sg docker -c "docker exec -d $CONTAINER bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=$N150_MESH_DESC
export TT_VISIBLE_DEVICES=$LLAMA_DEVICES
export MESH_DEVICE=N150
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
cd /home/container_app_user/voice-assistant
python main.py > /tmp/main_app.log 2>&1
'"

echo ""
echo "=========================================="
echo "  All services starting! Waiting for warmup..."
echo "=========================================="
echo ""
echo "  Device layout (Wormhole Galaxy):"
echo "    Device 0: Llama 3.1-8B (N150)"
echo "    Device 2: Whisper distil-large-v3 (N150, traced)"
echo "    Device 4: SpeechT5 TTS HOST (N150, traced, speaker 7306 female)"
echo "    Device 6: SpeechT5 TTS GUEST (N150, traced, speaker 1138 male)"
echo ""

# Health check loop
MAX_WAIT=300
ELAPSED=0
INTERVAL=10
ALL_READY=false

WHISPER_UP=false
TTS_HOST_UP=false
TTS_GUEST_UP=false
LLAMA_UP=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))

    if ! $WHISPER_UP; then
        sg docker -c "docker exec $CONTAINER grep -q 'Listening on' /tmp/whisper_server.log" 2>/dev/null && WHISPER_UP=true
        sg docker -c "docker exec $CONTAINER test -S /tmp/whisper_server.sock" 2>/dev/null && WHISPER_UP=true
    fi
    if ! $TTS_HOST_UP; then
        sg docker -c "docker exec $CONTAINER grep -q 'READY - Listening' /tmp/tts_server.log" 2>/dev/null && TTS_HOST_UP=true
        sg docker -c "docker exec $CONTAINER test -S /tmp/tts_server.sock" 2>/dev/null && TTS_HOST_UP=true
    fi
    if ! $TTS_GUEST_UP; then
        sg docker -c "docker exec $CONTAINER grep -q 'READY - Listening' /tmp/tts_server_guest.log" 2>/dev/null && TTS_GUEST_UP=true
        sg docker -c "docker exec $CONTAINER test -S /tmp/tts_server_guest.sock" 2>/dev/null && TTS_GUEST_UP=true
    fi
    if ! $LLAMA_UP; then
        sg docker -c "docker exec $CONTAINER grep -q 'All services ready' /tmp/main_app.log" 2>/dev/null && LLAMA_UP=true
    fi

    WHISPER_S=$($WHISPER_UP && echo "UP" || echo "...")
    TTS_H=$($TTS_HOST_UP && echo "UP" || echo "...")
    TTS_G=$($TTS_GUEST_UP && echo "UP" || echo "...")
    LLAMA_S=$($LLAMA_UP && echo "UP" || echo "...")

    echo "  [${ELAPSED}s] Whisper:${WHISPER_S}  TTS-Host:${TTS_H}  TTS-Guest:${TTS_G}  Llama:${LLAMA_S}"

    if $WHISPER_UP && $TTS_HOST_UP && $TTS_GUEST_UP && $LLAMA_UP; then
        ALL_READY=true
        break
    fi
done

echo ""
if $ALL_READY; then
    echo "=========================================="
    echo "  ALL SERVICES READY! (${ELAPSED}s)"
    echo "=========================================="
    echo ""
    echo "  Enter TT-HOME: http://localhost:8080/"
    echo ""
else
    echo "=========================================="
    echo "  SOME SERVICES NOT READY (after ${ELAPSED}s)"
    echo "=========================================="
    $WHISPER_UP    || echo "  Whisper FAILED     - sg docker -c 'docker exec $CONTAINER tail -30 /tmp/whisper_server.log'"
    $TTS_HOST_UP   || echo "  TTS HOST FAILED    - sg docker -c 'docker exec $CONTAINER tail -30 /tmp/tts_server.log'"
    $TTS_GUEST_UP  || echo "  TTS GUEST FAILED   - sg docker -c 'docker exec $CONTAINER tail -30 /tmp/tts_server_guest.log'"
    $LLAMA_UP      || echo "  Llama FAILED       - sg docker -c 'docker exec $CONTAINER tail -30 /tmp/main_app.log'"
fi
echo ""
