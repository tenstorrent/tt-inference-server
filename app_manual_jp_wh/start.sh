#!/bin/bash
# TT-HOME Voice Assistant - Wormhole Galaxy (Japanese, ELYZA-JP)
# Usage: ./start.sh
#
# ===== Architecture =====
# Hardware: Wormhole Galaxy (isolated N150 view per model)
# Docker: Lean tt-metal base (no weights) + weights mounted from host
#
# Device allocation (Galaxy N150 slices — skip odd chips):
#   ELYZA-JP-8B:         N150 (1 chip, device 0)
#   Whisper large-v3:    N150 (1 chip, device 2) - Japanese ASR
#   Qwen3 TTS (GUEST):  N150 (1 chip, device 4) - Jim voice
#   Qwen3 TTS (HOST):   N150 (1 chip, device 6) - Riata voice (podcast)
#
# Prerequisites:
#   1. Docker image: ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:manual_jp_wh
#      (tt-metal @ 747b4f4a63ef + ffmpeg/voice packages + WH JIT caches; NO HF weights)
#   2. Model weights downloaded to $WEIGHTS_DIR/hf_cache/
#      - elyza/Llama-3-ELYZA-JP-8B
#      - Qwen/Qwen3-TTS-12Hz-1.7B-Base
#      - openai/whisper-large-v3
#   See README.md for download commands.
# ==============================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="tt-wh-glxy-elyza-jp-manual"
IMAGE="${IMAGE:-ghcr.io/tenstorrent/tt-inference-server/tt-voice-assistant:manual_jp_wh}"
LOCAL_APP="$SCRIPT_DIR"

# Host path to model weights (HF cache)
WEIGHTS_DIR="${WEIGHTS_DIR:-$HOME/tt-home-weights}"

if [ ! -d "$WEIGHTS_DIR/hf_cache" ]; then
    echo "ERROR: Weights not found at $WEIGHTS_DIR/hf_cache"
    echo "Please download model weights first. See README.md for instructions."
    exit 1
fi

N150_MESH_DESC="/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/n150_mesh_graph_descriptor.textproto"

LLAMA_DEVICES="0"
WHISPER_DEVICES="2"
TTS_DEVICES="4"
TTS_HOST_DEVICES="6"

# Reference audio for voice cloning (GUEST = Jim, HOST = Riata)
REF_AUDIO="/home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/jim_reference.wav"
REF_TEXT_FILE="/home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/jim_reference.txt"
HOST_REF_AUDIO="/home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/female_reference_24k.wav"
HOST_REF_TEXT_FILE="/home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/female_reference.txt"

echo "=========================================="
echo "  TT-HOME Voice Assistant (Wormhole Galaxy - ELYZA Japan)"
echo "  TTS: Qwen3-TTS (TTNN) | LLM: ELYZA-JP-8B (Japanese)"
echo "=========================================="

# 1. Check if container exists, create if not
echo ""
echo "[1/6] Checking container..."
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "    Container not found, creating..."
    docker run -d --name $CONTAINER \
        --device /dev/tenstorrent:/dev/tenstorrent \
        -v /dev/hugepages-1G:/dev/hugepages-1G \
        -v "$SCRIPT_DIR":"$SCRIPT_DIR" \
        -v "$WEIGHTS_DIR/hf_cache":/home/container_app_user/.cache/huggingface/hub \
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

# Host-mounted HF cache is owned by the host user; container is uid 1000.
# Tokenizer load needs write access for .locks / refs (this is why TTS died first run).
chmod -R a+rwX "$WEIGHTS_DIR/hf_cache" || true
docker exec -u root $CONTAINER bash -lc 'mkdir -p /home/container_app_user/.cache/huggingface/hub/.locks && chmod -R a+rwX /home/container_app_user/.cache/huggingface/hub' || true

# 2. Stop any existing processes inside container
echo "[2/6] Stopping existing processes..."
docker exec $CONTAINER pkill -f "whisper_server.py" 2>/dev/null || true
docker exec $CONTAINER pkill -f "qwen3_tts_server.py" 2>/dev/null || true
docker exec $CONTAINER pkill -f "main.py" 2>/dev/null || true
docker exec $CONTAINER rm -f /tmp/tts_server.sock /tmp/tts_server_guest.sock /tmp/whisper_server.sock 2>/dev/null || true
sleep 2

# 3. Install dependencies & copy code
echo "[3/6] Installing dependencies and copying code..."

docker exec $CONTAINER mkdir -p /home/container_app_user/voice-assistant/services /home/container_app_user/voice-assistant/servers /home/container_app_user/voice-assistant/templates /home/container_app_user/voice-assistant/static /home/container_app_user/voice-assistant/output /home/container_app_user/voice-assistant/logs

docker cp $LOCAL_APP/main.py $CONTAINER:/home/container_app_user/voice-assistant/main.py
docker cp $LOCAL_APP/services/. $CONTAINER:/home/container_app_user/voice-assistant/services/
docker cp $LOCAL_APP/templates/. $CONTAINER:/home/container_app_user/voice-assistant/templates/
docker cp $LOCAL_APP/servers/. $CONTAINER:/home/container_app_user/voice-assistant/servers/
docker cp $LOCAL_APP/static/. $CONTAINER:/home/container_app_user/voice-assistant/static/
docker exec $CONTAINER mkdir -p /home/container_app_user/voice-assistant/output

# 4. Copy Qwen3 TTS model code into container (WH N150 tree we ran on this box)
echo "[4/6] Copying Qwen3 TTS model code (WH N150)..."
# Overlay copy only — do not rm. docker cp writes as root, so chown after.
docker exec -u root $CONTAINER mkdir -p /home/container_app_user/tt-metal/models/demos/qwen3_tts
docker cp $LOCAL_APP/qwen3_tts_latest/. $CONTAINER:/home/container_app_user/tt-metal/models/demos/qwen3_tts/
docker exec -u root $CONTAINER chown -R container_app_user:container_app_user /home/container_app_user/tt-metal/models/demos/qwen3_tts
echo "    Qwen3 TTS model code copied (WH-patched opti tree)"

# 5. Start Whisper + Qwen3 TTS servers
echo "[5/6] Starting model servers..."

echo "       Whisper large-v3 (Device $WHISPER_DEVICES, N150, Japanese)..."
docker exec -d $CONTAINER bash -c "
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=$N150_MESH_DESC
export TT_VISIBLE_DEVICES=$WHISPER_DEVICES
export MESH_DEVICE=N150
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/container_app_user/tt-metal
python /home/container_app_user/voice-assistant/servers/whisper_server.py --model openai/whisper-large-v3 --language ja > /tmp/whisper_server.log 2>&1
"

echo "       Qwen3 TTS GUEST (Device $TTS_DEVICES, N150, Jim voice)..."
docker exec -d $CONTAINER bash -c "
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=$N150_MESH_DESC
export TT_VISIBLE_DEVICES=$TTS_DEVICES
export MESH_DEVICE=N150
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /home/container_app_user/tt-metal
REF_TEXT=\$(cat $REF_TEXT_FILE)
python /home/container_app_user/voice-assistant/servers/qwen3_tts_server.py \
    --ref-audio $REF_AUDIO \
    --ref-text \"\$REF_TEXT\" \
    --socket /tmp/tts_server.sock \
    --device-id 0 > /tmp/tts_server.log 2>&1
"

echo "       Qwen3 TTS HOST (Device $TTS_HOST_DEVICES, N150, Riata voice)..."
docker exec -d $CONTAINER bash -c "
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=$N150_MESH_DESC
export TT_VISIBLE_DEVICES=$TTS_HOST_DEVICES
export MESH_DEVICE=N150
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
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

# 6. Start Main App (Llama on N150)
echo "[6/6] Starting Main App (ELYZA-JP-8B on Device $LLAMA_DEVICES, N150)..."
docker exec -d $CONTAINER bash -c "
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH=\"/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:\$PYTHONPATH\"
export TT_MESH_GRAPH_DESC_PATH=$N150_MESH_DESC
export TT_VISIBLE_DEVICES=$LLAMA_DEVICES
export MESH_DEVICE=N150
export HF_MODEL=elyza/Llama-3-ELYZA-JP-8B
cd /home/container_app_user/voice-assistant
python main.py > /tmp/main_app.log 2>&1
"

echo ""
echo "=========================================="
echo "  All services starting! Waiting for warmup..."
echo "=========================================="
echo ""
echo "  Device layout (Wormhole Galaxy, N150 per model):"
echo "    Device 0: ELYZA-JP-8B (N150)"
echo "    Device 2: Whisper large-v3 (N150, Japanese)"
echo "    Device 4: Qwen3-TTS GUEST - Jim voice (N150)"
echo "    Device 6: Qwen3-TTS HOST  - Riata voice (N150)"
echo ""

# Health check loop
MAX_WAIT=1200
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
        docker exec $CONTAINER test -S /tmp/whisper_server.sock 2>/dev/null && WHISPER_UP=true
    fi
    if ! $TTS_UP; then
        docker exec $CONTAINER test -S /tmp/tts_server.sock 2>/dev/null && TTS_UP=true
    fi
    if ! $TTS_HOST_UP; then
        docker exec $CONTAINER test -S /tmp/tts_server_guest.sock 2>/dev/null && TTS_HOST_UP=true
    fi
    if ! $LLAMA_UP; then
        docker exec $CONTAINER grep -q 'All services ready' /tmp/main_app.log 2>/dev/null && LLAMA_UP=true
    fi

    WHISPER_S=$($WHISPER_UP && echo "UP" || echo "...")
    TTS_S=$($TTS_UP && echo "UP" || echo "...")
    TTS_H=$($TTS_HOST_UP && echo "UP" || echo "...")
    LLAMA_S=$($LLAMA_UP && echo "UP" || echo "...")

    echo "  [${ELAPSED}s] Whisper:${WHISPER_S}  TTS-Jim:${TTS_S}  TTS-Riata:${TTS_H}  Llama:${LLAMA_S}"

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
    $LLAMA_UP     || echo "  Llama FAILED       - docker exec $CONTAINER tail -30 /tmp/main_app.log"
fi
echo ""
