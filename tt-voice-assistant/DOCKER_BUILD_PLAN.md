# Docker Build Plan - TT Voice Assistant

## Goal
Single command to run the voice assistant:
```bash
docker run -d --name tt-voice \
  --device /dev/tenstorrent:/dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -p 8080:8080 \
  tt-voice-assistant:final
```
Then open http://localhost:8080/ and it works.

---

## Current State
- Base image: `tt-voice-assistant:push-button` or `tt-voice-assistant:clean-single-layer`
- Working code: `/home/ttuser/teja/tt-voice-assistant/app/`
- Model caches exist in image at `/home/container_app_user/tt-metal/model_cache/`

## What Needs to be Added to Dockerfile

### 1. System Dependencies
```dockerfile
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
```

### 2. Python Dependencies
```dockerfile
RUN pip install onnx
```

### 3. Environment Variables (already in image, verify)
```dockerfile
ENV PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal"
# NOTE: TT_MESH_GRAPH_DESC_PATH is NOT set globally - it's set per model in entrypoint
```

### 4. Copy Application Code
```dockerfile
COPY app/ /home/container_app_user/voice-assistant/
```

Files to copy:
- `main_fixed.py` → `main.py`
- `services/` (all socket clients)
- `servers/` (face_auth, whisper, speecht5/qwen3 tts)
- `templates/` (index.html, index_chat.html)
- `static/` (if any)
- `registered_faces/` (pre-registered faces)

### 5. Entrypoint Script
Create `/home/container_app_user/voice-assistant/entrypoint.sh`:
```bash
#!/bin/bash
set -e

MESH_DESC=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto

# Start Face Auth Server (Device 0)
TT_MESH_GRAPH_DESC_PATH=$MESH_DESC TT_VISIBLE_DEVICES=0 \
  python /home/container_app_user/voice-assistant/servers/face_auth_server.py &

# Start Whisper Server (Device 2)
TT_MESH_GRAPH_DESC_PATH=$MESH_DESC TT_VISIBLE_DEVICES=2 \
  python /home/container_app_user/voice-assistant/servers/whisper_server.py &

# Start TTS Server (CPU - SpeechT5, no TT device needed)
python /home/container_app_user/voice-assistant/servers/speecht5_server.py --speaker-id 1138 &

# Wait for servers to initialize
sleep 15

# Start Main App (Llama on Device 1)
TT_MESH_GRAPH_DESC_PATH=$MESH_DESC TT_VISIBLE_DEVICES=1 \
  python /home/container_app_user/voice-assistant/main.py
```

```dockerfile
COPY entrypoint.sh /home/container_app_user/voice-assistant/
RUN chmod +x /home/container_app_user/voice-assistant/entrypoint.sh
ENTRYPOINT ["/home/container_app_user/voice-assistant/entrypoint.sh"]
```

### 6. Working Directory
```dockerfile
WORKDIR /home/container_app_user/tt-metal
```
(Important for Llama model cache path to be found)

### 7. Expose Port
```dockerfile
EXPOSE 8080
```

---

## Device Mapping
| Model | TT_VISIBLE_DEVICES | Notes |
|-------|-------------------|-------|
| Face Auth (YuNet+SFace) | 0 | |
| Llama 3.1 8B | 1 | P150 single device |
| Whisper | 2 | |
| TTS (SpeechT5) | CPU | No TT device needed |

---

## Model Caches (Already in Image)
- `/home/container_app_user/tt-metal/model_cache/meta-llama/Llama-3.1-8B-Instruct/P150/` - Llama cache
- Whisper cache - needs verification
- YuNet/SFace - downloads on first run to `/root/.cache/ttnn/models/`

---

## Build Steps

1. **Create Dockerfile** in `/home/ttuser/teja/tt-voice-assistant/`
2. **Create entrypoint.sh** 
3. **Build image**:
   ```bash
   docker build -t tt-voice-assistant:final .
   ```
4. **Test**:
   ```bash
   docker run -d --name tt-voice-final \
     --device /dev/tenstorrent:/dev/tenstorrent \
     -v /dev/hugepages-1G:/dev/hugepages-1G \
     -p 8080:8080 \
     tt-voice-assistant:final
   ```

---

## Runtime Flags (User Must Provide)
- `--device /dev/tenstorrent:/dev/tenstorrent` - TT hardware access
- `-v /dev/hugepages-1G:/dev/hugepages-1G` - Hugepages for TT Metal
- `-p 8080:8080` - Web UI port

## No Runtime Flags Needed For
- `PYTHONPATH` - baked into image
- `TT_MESH_GRAPH_DESC_PATH` - set per-process in entrypoint (not global)
- `TT_VISIBLE_DEVICES` - set per-process in entrypoint

---

## Optional: TTS Choice
To use Qwen3 TTS instead of SpeechT5, change entrypoint to run:
```bash
TT_VISIBLE_DEVICES=3 python /home/container_app_user/voice-assistant/servers/tts_server.py &
```
(Qwen3 uses TT device 3)

---

## Files Checklist
- [ ] Dockerfile
- [ ] entrypoint.sh
- [ ] app/main_fixed.py → main.py
- [ ] app/services/*.py
- [ ] app/servers/*.py
- [ ] app/templates/*.html
- [ ] app/registered_faces/
