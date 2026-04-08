# TT Voice Assistant - Run Guide

## Local Code Location

All code is in: `/home/ttuser/teja/tt-voice-assistant/app/`

```
app/
├── main.py                      # Original FastAPI app (uses model_manager)
├── main_fixed.py                # Socket-based FastAPI app (uses socket servers)
├── services/
│   ├── model_manager.py         # Orchestrates all models (for main.py)
│   ├── face_auth_service.py     # Direct face auth (for main.py)
│   ├── whisper_service.py       # Direct whisper (for main.py)
│   ├── llama_service.py         # Direct llama (for main.py)
│   ├── tts_service.py           # Direct TTS - Qwen3 (for main.py)
│   ├── face_auth_service_socket.py  # Socket client (for main_fixed.py)
│   ├── whisper_service_socket.py    # Socket client (for main_fixed.py)
│   ├── tts_service_socket.py        # Socket client - SpeechT5 (for main_fixed.py)
│   └── llama_service_socket.py      # Socket client (for main_fixed.py)
├── servers/
│   ├── face_auth_server.py      # YuNet+SFace on TT Device 0
│   ├── whisper_server.py        # Whisper on TT Device 2
│   └── speecht5_server.py       # SpeechT5 TTS on CPU
├── templates/
│   ├── index.html               # Old UI
│   └── index_chat.html          # New chat UI
├── static/
└── registered_faces/            # Face embeddings (Teja, Mohamed, etc.)
```

---

## Quick Start

### 1. Start Container

```bash
docker run -d \
  --name tt-voice-test \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v /home/ttuser/teja/tt-voice-assistant:/home/ttuser/teja/tt-voice-assistant \
  -p 8080:8080 \
  tt-voice-assistant:push-button \
  sleep infinity
```

Or restart existing:
```bash
docker start tt-voice-test
```

### 2. Install ffmpeg (one time after container start)

```bash
docker exec tt-voice-test apt-get update && docker exec tt-voice-test apt-get install -y ffmpeg
```

### 3. Copy Files to Container

```bash
# Copy registered faces
docker cp /home/ttuser/teja/tt-voice-assistant/app/registered_faces/. \
  tt-voice-test:/home/container_app_user/voice-assistant/registered_faces/

# Copy new chat UI template
docker cp /home/ttuser/teja/tt-voice-assistant/app/templates/index_chat.html \
  tt-voice-test:/home/container_app_user/voice-assistant/templates/

# Add /chat route to main.py (if not exists)
docker exec tt-voice-test bash -c 'grep -q "def get_chat_page" /home/container_app_user/voice-assistant/main.py || cat >> /home/container_app_user/voice-assistant/main.py << "EOF"

@app.get("/chat", response_class=HTMLResponse)
async def get_chat_page():
    return FileResponse("/home/container_app_user/voice-assistant/templates/index_chat.html")
EOF'
```

### 4. Start the App

**Option A: Use Docker image's built-in main.py (Qwen3 TTS)**
```bash
docker exec -d tt-voice-test bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH"
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=1
cd /home/container_app_user/voice-assistant
python main.py > /tmp/main_app.log 2>&1
'
```

**Option B: Use socket servers (SpeechT5 TTS - faster)**

First start the servers:
```bash
# Face Auth (Device 0)
docker exec -d tt-voice-test bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH"
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=0
cd /home/container_app_user/tt-metal
python /home/ttuser/teja/tt-voice-assistant/app/servers/face_auth_server.py > /tmp/face_auth_server.log 2>&1
'

# Whisper (Device 2)
docker exec -d tt-voice-test bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH"
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=2
cd /home/container_app_user/tt-metal
python /home/ttuser/teja/tt-voice-assistant/app/servers/whisper_server.py > /tmp/whisper_server.log 2>&1
'

# TTS - SpeechT5 (CPU)
docker exec -d tt-voice-test bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH"
cd /home/container_app_user/tt-metal
python /home/ttuser/teja/tt-voice-assistant/app/servers/speecht5_server.py --speaker-id 1138 > /tmp/tts_server.log 2>&1
'
```

Then copy and start main_fixed.py:
```bash
docker cp /home/ttuser/teja/tt-voice-assistant/app/main_fixed.py \
  tt-voice-test:/home/container_app_user/voice-assistant/main.py

docker cp /home/ttuser/teja/tt-voice-assistant/app/services/. \
  tt-voice-test:/home/container_app_user/voice-assistant/services/

docker exec -d tt-voice-test bash -c '
source /home/container_app_user/tt-metal/python_env/bin/activate
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH"
export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=1
cd /home/container_app_user/voice-assistant
python main.py > /tmp/main_app.log 2>&1
'
```

### 5. Access UI

- **http://localhost:8080/** - Old UI
- **http://localhost:8080/chat** - New chat UI

---

## Check Logs

```bash
docker exec tt-voice-test tail -20 /tmp/main_app.log
docker exec tt-voice-test tail -20 /tmp/face_auth_server.log
docker exec tt-voice-test tail -20 /tmp/whisper_server.log
docker exec tt-voice-test tail -20 /tmp/tts_server.log
```

---

## Kill/Restart

```bash
# Kill all python processes
docker exec tt-voice-test pkill -f python

# Restart container
docker restart tt-voice-test
```

---

## Device Allocation

| Device | Model |
|--------|-------|
| 0 | Face Auth (YuNet + SFace) |
| 1 | Llama 3.1 8B Instruct |
| 2 | Whisper (distil-large-v3) |
| CPU | SpeechT5 TTS (Option B) or Qwen3 TTS (Option A) |

---

## Notes

- All code lives locally in `/home/ttuser/teja/tt-voice-assistant/app/`
- Copy to container before running
- `PYTHONPATH` must include `/usr/local/lib/python3.10/dist-packages` for onnx
- Face auth threshold is 0.5
- ffmpeg needed for audio conversion (install once per container)
