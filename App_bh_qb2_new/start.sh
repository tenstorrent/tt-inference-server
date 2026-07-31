#!/bin/bash
# TT-HOME Voice Assistant - Blackhole QB2 (Qwen3 TTS + Gemma4-12B)
# Usage: ./start.sh
#
# ===== Architecture =====
# Hardware: Blackhole QB2 (4x P150 chips)
# Docker:  tt-voice-assistant:new_slim  (Gemma4 HF+TT baked, Llama removed)
#
# Device allocation:
#   Gemma4-12B:         P150 (1 chip, device 0)
#   Whisper distil-v3:  P150 (1 chip, device 1) - traced
#   Qwen3 TTS (GUEST):  P150 (1 chip, device 2) - Jim voice
#   Qwen3 TTS (HOST):   P150 (1 chip, device 3) - Riata voice (podcast)
#
# Weights: baked into :new_slim (no host HF mounts required).
# Optional: USE_HOST_WEIGHTS=1 to bind-mount host caches (dev only).
# ==============================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="${CONTAINER:-tt-bh-qb2-new-slim}"
IMAGE="${IMAGE:-ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:new_slim}"
LOCAL_APP="$SCRIPT_DIR"
USE_HOST_WEIGHTS="${USE_HOST_WEIGHTS:-0}"

HOST_HF_HOME="${HOST_HF_HOME:-$HOME/.cache/huggingface}"
HOST_HF_HUB="${HOST_HF_HUB:-$HOST_HF_HOME/hub}"
HOST_TT_CACHE_ROOT="${HOST_TT_CACHE_ROOT:-$HOST_HF_HOME/tt_cache}"
HF_MODEL="${HF_MODEL:-google/gemma-4-12B-it}"
TT_CACHE_PATH_CTR="/home/container_app_user/.cache/huggingface/tt_cache/google--gemma-4-12B-it"

P150_MESH_DESC="/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"

LLAMA_DEVICES="0"
WHISPER_DEVICES="1"
TTS_DEVICES="2"
TTS_HOST_DEVICES="3"

REF_AUDIO="/home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/jim_reference.wav"
REF_TEXT_FILE="/home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/jim_reference.txt"
HOST_REF_AUDIO="/home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/female_reference_24k.wav"
HOST_REF_TEXT_FILE="/home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/female_reference.txt"

echo "=========================================="
echo "  TT-HOME Voice Assistant (Blackhole QB2)"
echo "  TTS: Qwen3-TTS (TTNN) | LLM: Gemma4-12B"
echo "  Image: $IMAGE"
echo "=========================================="

# 1. Check if container exists, create if not
echo ""
echo "[1/6] Checking container..."
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "    Creating container from baked image..."
    RUN_ARGS=(
        -d --name "$CONTAINER"
        --device /dev/tenstorrent:/dev/tenstorrent
        -v /dev/hugepages-1G:/dev/hugepages-1G
        -v "$SCRIPT_DIR":"$SCRIPT_DIR"
        -p 8080:8080
        --entrypoint /bin/bash
    )
    if [ "$USE_HOST_WEIGHTS" = "1" ]; then
        echo "    USE_HOST_WEIGHTS=1 — mounting host HF hub + TT cache"
        RUN_ARGS+=(
            -v "$HOST_HF_HUB":/home/container_app_user/.cache/huggingface/hub
            -v "$HOST_TT_CACHE_ROOT":/home/container_app_user/.cache/huggingface/tt_cache
        )
    fi
    docker run "${RUN_ARGS[@]}" "$IMAGE" -c "sleep infinity"
    echo "    Container created"
    sleep 5
elif ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "    Container exists but stopped, starting..."
    docker start "$CONTAINER"
    sleep 3
else
    echo "    Container already running"
fi

# Verify baked Gemma weights (unless host-mounted)
if [ "$USE_HOST_WEIGHTS" != "1" ]; then
    if ! docker exec "$CONTAINER" test -d /home/container_app_user/.cache/huggingface/hub/models--google--gemma-4-12B-it; then
        echo "ERROR: Gemma4 HF weights missing in image. Rebuild with ./build_new_slim.sh"
        exit 1
    fi
    if ! docker exec "$CONTAINER" test -d "$TT_CACHE_PATH_CTR"; then
        echo "ERROR: Gemma4 TT cache missing in image. Rebuild with ./build_new_slim.sh"
        exit 1
    fi
fi

# 2. Stop any existing processes inside container
echo "[2/6] Stopping existing processes..."
docker exec "$CONTAINER" pkill -f "whisper_server.py" 2>/dev/null || true
docker exec "$CONTAINER" pkill -f "qwen3_tts_server.py" 2>/dev/null || true
docker exec "$CONTAINER" pkill -f "main.py" 2>/dev/null || true
docker exec "$CONTAINER" rm -f /tmp/tts_server.sock /tmp/tts_server_guest.sock /tmp/whisper_server.sock 2>/dev/null || true
sleep 2

# 3. Sync local app code (optional hot-update; image already has a baked copy)
echo "[3/6] Syncing app code from $LOCAL_APP ..."
docker cp "$LOCAL_APP/main.py" "$CONTAINER:/home/container_app_user/voice-assistant/main.py"
docker cp "$LOCAL_APP/services/." "$CONTAINER:/home/container_app_user/voice-assistant/services/"
docker cp "$LOCAL_APP/templates/." "$CONTAINER:/home/container_app_user/voice-assistant/templates/"
docker cp "$LOCAL_APP/servers/." "$CONTAINER:/home/container_app_user/voice-assistant/servers/"
docker cp "$LOCAL_APP/static/." "$CONTAINER:/home/container_app_user/voice-assistant/static/"
docker exec "$CONTAINER" mkdir -p /home/container_app_user/voice-assistant/output /home/container_app_user/voice-assistant/logs

if [ -d "$LOCAL_APP/qwen3_tts_latest" ]; then
    docker exec "$CONTAINER" rm -rf /home/container_app_user/tt-metal/models/demos/qwen3_tts
    docker exec "$CONTAINER" mkdir -p /home/container_app_user/tt-metal/models/demos/qwen3_tts
    docker cp "$LOCAL_APP/qwen3_tts_latest/." "$CONTAINER:/home/container_app_user/tt-metal/models/demos/qwen3_tts/"
fi

# 4. Start Whisper + Qwen3 TTS servers
echo "[4/6] Starting model servers..."

echo "       Whisper (Device $WHISPER_DEVICES, P150, traced)..."
docker exec -d "$CONTAINER" bash -c "
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=$P150_MESH_DESC
export TT_VISIBLE_DEVICES=$WHISPER_DEVICES
cd /home/container_app_user/tt-metal
python /home/container_app_user/voice-assistant/servers/whisper_server.py > /tmp/whisper_server.log 2>&1
"

echo "       Qwen3 TTS GUEST (Device $TTS_DEVICES, P150, Jim voice)..."
docker exec -d "$CONTAINER" bash -c "
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=$P150_MESH_DESC
export TT_VISIBLE_DEVICES=$TTS_DEVICES
cd /home/container_app_user/tt-metal
REF_TEXT=\$(cat $REF_TEXT_FILE)
python /home/container_app_user/voice-assistant/servers/qwen3_tts_server.py \
    --ref-audio $REF_AUDIO \
    --ref-text \"\$REF_TEXT\" \
    --socket /tmp/tts_server.sock \
    --device-id 0 > /tmp/tts_server.log 2>&1
"

echo "       Qwen3 TTS HOST (Device $TTS_HOST_DEVICES, P150, Riata voice)..."
docker exec -d "$CONTAINER" bash -c "
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=$P150_MESH_DESC
export TT_VISIBLE_DEVICES=$TTS_HOST_DEVICES
cd /home/container_app_user/tt-metal
HOST_TEXT=\$(cat $HOST_REF_TEXT_FILE)
python /home/container_app_user/voice-assistant/servers/qwen3_tts_server.py \
    --ref-audio $HOST_REF_AUDIO \
    --ref-text \"\$HOST_TEXT\" \
    --socket /tmp/tts_server_guest.sock \
    --device-id 0 \
    --temperature 0.7 > /tmp/tts_server_guest.log 2>&1
"

echo "       Waiting for servers to initialize..."
sleep 15

# 5. Start Main App (Gemma4 on P150)
echo "[5/6] Starting Main App (Gemma4-12B on Device $LLAMA_DEVICES, P150)..."
docker exec -d "$CONTAINER" bash -c "
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=$P150_MESH_DESC
export TT_VISIBLE_DEVICES=$LLAMA_DEVICES
export MESH_DEVICE=P150
export HF_HOME=/home/container_app_user/.cache/huggingface
export HF_HUB_CACHE=/home/container_app_user/.cache/huggingface/hub
export HF_MODEL=$HF_MODEL
export TT_CACHE_PATH=$TT_CACHE_PATH_CTR
export GEMMA4_MAX_SEQ_LEN=\${GEMMA4_MAX_SEQ_LEN:-4096}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/container_app_user/voice-assistant
python main.py > /tmp/main_app.log 2>&1
"

echo ""
echo "=========================================="
echo "  All services starting! Waiting for warmup..."
echo "=========================================="
echo ""
echo "  Device layout (Blackhole QB2):"
echo "    Device 0: Gemma4-12B (P150, baked HF+TT cache)"
echo "    Device 1: Whisper distil-large-v3 (P150, traced)"
echo "    Device 2: Qwen3-TTS GUEST - Jim voice (P150)"
echo "    Device 3: Qwen3-TTS HOST  - Riata voice (P150)"
echo ""

# 6. Health check
MAX_WAIT=600
ELAPSED=0
INTERVAL=10
ALL_READY=false

WHISPER_UP=false
TTS_UP=false
TTS_HOST_UP=false
LLAMA_UP=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))

    if ! $WHISPER_UP; then
        docker exec "$CONTAINER" test -S /tmp/whisper_server.sock 2>/dev/null && WHISPER_UP=true
    fi
    if ! $TTS_UP; then
        docker exec "$CONTAINER" test -S /tmp/tts_server.sock 2>/dev/null && TTS_UP=true
    fi
    if ! $TTS_HOST_UP; then
        docker exec "$CONTAINER" test -S /tmp/tts_server_guest.sock 2>/dev/null && TTS_HOST_UP=true
    fi
    if ! $LLAMA_UP; then
        docker exec "$CONTAINER" grep -q 'All services ready' /tmp/main_app.log 2>/dev/null && LLAMA_UP=true
    fi

    WHISPER_S=$($WHISPER_UP && echo "UP" || echo "...")
    TTS_S=$($TTS_UP && echo "UP" || echo "...")
    TTS_H=$($TTS_HOST_UP && echo "UP" || echo "...")
    LLAMA_S=$($LLAMA_UP && echo "UP" || echo "...")

    echo "  [${ELAPSED}s] Whisper:${WHISPER_S}  TTS-Jim:${TTS_S}  TTS-Riata:${TTS_H}  Gemma4:${LLAMA_S}"

    if $WHISPER_UP && $TTS_UP && $TTS_HOST_UP && $LLAMA_UP; then
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
    $WHISPER_UP   || echo "  Whisper FAILED     - docker exec $CONTAINER tail -30 /tmp/whisper_server.log"
    $TTS_UP       || echo "  TTS-Jim FAILED     - docker exec $CONTAINER tail -30 /tmp/tts_server.log"
    $TTS_HOST_UP  || echo "  TTS-Riata FAILED   - docker exec $CONTAINER tail -30 /tmp/tts_server_guest.log"
    $LLAMA_UP     || echo "  Gemma4 FAILED      - docker exec $CONTAINER tail -50 /tmp/main_app.log"
fi
echo ""
