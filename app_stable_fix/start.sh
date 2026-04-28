#!/bin/bash
# TT Voice Assistant - Single command to start everything
# Usage: ./start.sh
#
# ===== TTS BACKEND SWITCH =====
# To use Lightning V2:  TTS_BACKEND=lv2       (default)
# To use SpeechT5:      TTS_BACKEND=speecht5
#
# Quick switch: change the line below and re-run ./start.sh
TTS_BACKEND="${TTS_BACKEND:-lv2}"
# ==============================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="tt-voice-test"
IMAGE="ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:push-button"
LOCAL_APP="$SCRIPT_DIR"

echo "=========================================="
echo "  TT Voice Assistant - Starting Pipeline"
echo "  TTS Backend: $TTS_BACKEND"
echo "=========================================="

# 0. Pre-flight: if using LV2, check that lv2-tts container is running
if [ "$TTS_BACKEND" = "lv2" ]; then
    echo ""
    echo "[0] Checking Lightning V2 TTS container..."
    if docker ps --format '{{.Names}}' | grep -q "^lv2-tts$"; then
        echo "    lv2-tts is running ✅"
        curl -sf http://localhost:2082/speakers > /dev/null 2>&1 \
            && echo "    TTS API responding ✅" \
            || echo "    ⚠️  TTS API not ready yet (may still be warming up)"
    else
        echo "    ❌ lv2-tts container is NOT running!"
        echo "    Start it first:  cd /home/ttuser/ssinghal/lv2 && bash run_docker1"
        exit 1
    fi
fi

# 1. Check if container exists, create if not
echo ""
echo "[1/6] Checking container..."
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "    Container not found, creating..."
    docker run -d --name $CONTAINER \
        --device /dev/tenstorrent:/dev/tenstorrent \
        -v /dev/hugepages-1G:/dev/hugepages-1G \
        -v "$SCRIPT_DIR":"$SCRIPT_DIR" \
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
docker exec $CONTAINER pkill -f "speecht5_ttnn_server.py" 2>/dev/null || true
docker exec $CONTAINER pkill -f "tts_server.py" 2>/dev/null || true
docker exec $CONTAINER pkill -f "main.py" 2>/dev/null || true
sleep 2

# 3. Install dependencies
echo "[3/6] Installing dependencies..."
docker exec $CONTAINER bash -c 'apt-get update -qq && apt-get install -y -qq ffmpeg 2>/dev/null || true'
docker exec $CONTAINER bash -c 'pip install onnx PyPDF2 requests beautifulsoup4 -q 2>/dev/null || true'

# 4. Copy latest code to container
echo "[4/6] Copying latest code to container..."
docker cp $LOCAL_APP/main_fixed.py $CONTAINER:/home/container_app_user/voice-assistant/main.py
docker cp $LOCAL_APP/services/. $CONTAINER:/home/container_app_user/voice-assistant/services/
docker cp $LOCAL_APP/templates/. $CONTAINER:/home/container_app_user/voice-assistant/templates/
docker cp $LOCAL_APP/servers/. $CONTAINER:/home/container_app_user/voice-assistant/servers/
docker cp $LOCAL_APP/static/. $CONTAINER:/home/container_app_user/voice-assistant/static/
docker cp $LOCAL_APP/registered_faces/. $CONTAINER:/home/container_app_user/voice-assistant/registered_faces/ 2>/dev/null || true
docker exec $CONTAINER mkdir -p /home/container_app_user/voice-assistant/output

# Patch SpeechT5 encoder sizes (needed even if using LV2, in case of fallback)
GENERATOR_PY="/home/container_app_user/tt-metal/models/experimental/speecht5_tts/tt/ttnn_speecht5_generator.py"
docker exec $CONTAINER sed -i 's/SUPPORTED_ENCODER_SEQ_LENS = \[128, 256, 384, 512, 768\]/SUPPORTED_ENCODER_SEQ_LENS = [64, 128, 256]/' "$GENERATOR_PY" 2>/dev/null || true

# ===== Device layout depends on TTS backend =====
if [ "$TTS_BACKEND" = "lv2" ]; then
    # LV2: Device 0 taken by external lv2-tts container
    #   Device 0: Lightning V2 (external)
    #   Device 1: Llama
    #   Device 2: Whisper
    #   Device 3: Face Auth
    FACE_DEVICE=3
    WHISPER_DEVICE=2
    LLAMA_DEVICE=1
else
    # SpeechT5: original layout
    #   Device 0: Face Auth
    #   Device 1: Llama
    #   Device 2: SpeechT5 TTS
    #   Device 3: Whisper
    FACE_DEVICE=0
    WHISPER_DEVICE=3
    LLAMA_DEVICE=1
fi

# 5. Start Face Auth + Whisper + (SpeechT5 if needed)
echo "[5/6] Starting model servers..."
echo "       Face Auth (Device $FACE_DEVICE)..."
docker exec -d $CONTAINER bash -c "
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=$FACE_DEVICE
cd /home/container_app_user/tt-metal
python /home/container_app_user/voice-assistant/servers/face_auth_server.py > /tmp/face_auth_server.log 2>&1
"

echo "       Whisper (Device $WHISPER_DEVICE)..."
docker exec -d $CONTAINER bash -c "
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=$WHISPER_DEVICE
cd /home/container_app_user/tt-metal
python /home/container_app_user/voice-assistant/servers/whisper_server.py > /tmp/whisper_server.log 2>&1
"

if [ "$TTS_BACKEND" = "speecht5" ]; then
    echo "       SpeechT5 TTS (Device 2)..."
    docker exec -d $CONTAINER bash -c '
    source /home/container_app_user/tt-metal/python_env/bin/activate
    export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH"
    export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
    export TT_VISIBLE_DEVICES=2
    cd /home/container_app_user/tt-metal
    python /home/container_app_user/voice-assistant/servers/speecht5_ttnn_server.py --speaker-id 7306 > /tmp/tts_server.log 2>&1
    '
fi

echo "       Waiting for servers to initialize..."
sleep 10

# 6. Start Main App (Llama)
echo "[6/6] Starting Main App (Llama on Device $LLAMA_DEVICE)..."
docker exec -d $CONTAINER bash -c "
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=$LLAMA_DEVICE
export TTS_BACKEND=$TTS_BACKEND
cd /home/container_app_user/voice-assistant
python main.py > /tmp/main_app.log 2>&1
"

echo ""
echo "=========================================="
echo "  All services starting! Waiting for warmup..."
echo "=========================================="
echo ""
if [ "$TTS_BACKEND" = "lv2" ]; then
    echo "  Device layout:"
    echo "    Device 0: Lightning V2 TTS (external lv2-tts container)"
    echo "    Device 1: Llama"
    echo "    Device 2: Whisper"
    echo "    Device 3: Face Auth"
else
    echo "  Device layout:"
    echo "    Device 0: Face Auth"
    echo "    Device 1: Llama"
    echo "    Device 2: SpeechT5 TTS"
    echo "    Device 3: Whisper"
fi
echo ""

# Health check loop
MAX_WAIT=300
ELAPSED=0
INTERVAL=10
ALL_READY=false

FACE_UP=false
WHISPER_UP=false
TTS_UP=false
LLAMA_UP=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))

    if ! $FACE_UP; then
        docker exec $CONTAINER grep -q "Listening on" /tmp/face_auth_server.log 2>/dev/null && FACE_UP=true
    fi
    if ! $FACE_UP; then
        docker exec $CONTAINER test -S /tmp/face_auth_server.sock 2>/dev/null && FACE_UP=true
    fi
    if ! $WHISPER_UP; then
        docker exec $CONTAINER grep -q "Listening on" /tmp/whisper_server.log 2>/dev/null && WHISPER_UP=true
    fi
    if ! $WHISPER_UP; then
        docker exec $CONTAINER test -S /tmp/whisper_server.sock 2>/dev/null && WHISPER_UP=true
    fi
    if ! $TTS_UP; then
        if [ "$TTS_BACKEND" = "lv2" ]; then
            curl -sf http://localhost:2082/speakers > /dev/null 2>&1 && TTS_UP=true
        else
            docker exec $CONTAINER grep -q "READY - Listening" /tmp/tts_server.log 2>/dev/null && TTS_UP=true
            docker exec $CONTAINER test -S /tmp/tts_server.sock 2>/dev/null && TTS_UP=true
        fi
    fi
    if ! $LLAMA_UP; then
        docker exec $CONTAINER grep -q "All services ready" /tmp/main_app.log 2>/dev/null && LLAMA_UP=true
    fi

    FACE_S=$($FACE_UP && echo "✅" || echo "⏳")
    WHISPER_S=$($WHISPER_UP && echo "✅" || echo "⏳")
    TTS_S=$($TTS_UP && echo "✅" || echo "⏳")
    LLAMA_S=$($LLAMA_UP && echo "✅" || echo "⏳")

    FACE_ERR=""
    TTS_ERR=""
    if ! $FACE_UP; then
        docker exec $CONTAINER grep -qE "failed to initialize|TT_FATAL|Segmentation fault" /tmp/face_auth_server.log 2>/dev/null && FACE_ERR=" ❌"
    fi
    if ! $TTS_UP && [ "$TTS_BACKEND" = "speecht5" ]; then
        docker exec $CONTAINER grep -qE "failed to initialize|TT_FATAL|Segmentation fault" /tmp/tts_server.log 2>/dev/null && TTS_ERR=" ❌"
    fi

    TTS_LABEL="TTS(LV2)"
    [ "$TTS_BACKEND" = "speecht5" ] && TTS_LABEL="TTS(ST5)"
    echo "  [${ELAPSED}s] Face:${FACE_S}${FACE_ERR}  Whisper:${WHISPER_S}  ${TTS_LABEL}:${TTS_S}${TTS_ERR}  Llama:${LLAMA_S}"

    if $FACE_UP && $WHISPER_UP && $TTS_UP && $LLAMA_UP; then
        ALL_READY=true
        break
    fi
done

echo ""
if $ALL_READY; then
    echo "=========================================="
    echo "  ✅ ALL SERVICES READY! (${ELAPSED}s)"
    echo "=========================================="
    echo ""
    if [ "$TTS_BACKEND" = "lv2" ]; then
        echo "  TTS Speakers:"
        curl -sf http://localhost:2082/speakers | python3 -c "import sys,json; [print(f'    - {s}') for s in json.load(sys.stdin)]" 2>/dev/null || echo "    (could not fetch speaker list)"
    fi
    echo ""
    echo "  Enter TT-HOME: http://localhost:8080/"
    echo ""
    echo "  💡 Tip: Use Google Chrome for the best experience!"
else
    echo "=========================================="
    echo "  ⚠️  SOME SERVICES FAILED (after ${ELAPSED}s)"
    echo "=========================================="
    $FACE_UP    || echo "  ❌ Face Auth — docker exec $CONTAINER tail -30 /tmp/face_auth_server.log"
    $WHISPER_UP || echo "  ❌ Whisper   — docker exec $CONTAINER tail -30 /tmp/whisper_server.log"
    if [ "$TTS_BACKEND" = "lv2" ]; then
        $TTS_UP || echo "  ❌ TTS (LV2) — check: docker logs lv2-tts --tail 30"
    else
        $TTS_UP || echo "  ❌ TTS (ST5) — docker exec $CONTAINER tail -30 /tmp/tts_server.log"
    fi
    $LLAMA_UP   || echo "  ❌ Llama     — docker exec $CONTAINER tail -30 /tmp/main_app.log"
fi
echo ""
