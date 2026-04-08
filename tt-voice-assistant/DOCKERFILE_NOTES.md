# Dockerfile Notes - Dependencies to Add

When building the final Docker image, add these to the Dockerfile:

## System Packages

```dockerfile
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

## Python Packages (already in venv, but ensure PYTHONPATH includes system packages)

```dockerfile
ENV PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH"
```

The `onnx` package is in `/usr/local/lib/python3.10/dist-packages` and needs to be accessible.

## Required for:
- **ffmpeg**: Audio format conversion (WebM/Opus from browser → WAV for Whisper)
- **onnx**: SFace model loading (face recognition)

## Current Working Setup

The image `tt-voice-assistant:push-button` needs these installed manually after container start:
```bash
docker exec tt-voice-test apt-get update && apt-get install -y ffmpeg
```
