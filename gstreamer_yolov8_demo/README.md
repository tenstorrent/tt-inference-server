# GStreamer YOLOv8s Demo on Tenstorrent

Real-time YOLOv8s object detection using native GStreamer pipeline on Tenstorrent hardware.

## Overview

This demo provides a **native GStreamer plugin** for YOLOv8s inference, similar to NVIDIA DeepStream. The model runs directly on Tenstorrent Wormhole devices at **100+ FPS** after warmup.

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              GStreamer Pipeline                       │  │
│  │  Input → decode → yolov8s (TT device) → MJPEG stream  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         ↑                                    ↓
    Video Input                         Browser/VLC
  (file/RTSP/webcam)                  http://host:8080
```

## Quick Start

### 1. Build Docker Image (one time)

```bash
cd gstreamer_yolov8_demo
docker build -t tt-gstreamer-yolov8 .
```

### 2. Run

**Test pattern (no camera needed):**
```bash
docker run --rm -p 8080:8080 \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  tt-gstreamer-yolov8 stream-test
```

**With video file:**
```bash
docker run --rm -p 8080:8080 \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v /path/to/videos:/videos \
  tt-gstreamer-yolov8 stream-file /videos/input.mp4
```

**With RTSP camera (CCTV):**
```bash
docker run --rm -p 8080:8080 \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  tt-gstreamer-yolov8 stream-rtsp rtsp://user:pass@192.168.1.100:554/stream
```

**With USB webcam:**
```bash
docker run --rm -p 8080:8080 \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --device /dev/video0:/dev/video0 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  tt-gstreamer-yolov8 stream-webcam
```

### 3. View Stream

- **Browser:** `http://HOST_IP:8080`
- **VLC:** `vlc tcp://HOST_IP:8080`

**Remote Access (via SSH tunnel):**
```bash
# From your local machine, forward port 8080:
ssh -L 8080:localhost:8080 user@remote-host

# Then open in your local browser:
# http://localhost:8080
```

> **Note:** First run takes ~2 minutes for model trace compilation. Subsequent frames run at full speed.

## Modes

| Mode | Description | Example |
|------|-------------|---------|
| `stream-test` | Synthetic test pattern | `stream-test` |
| `stream-file` | Local video file | `stream-file /videos/input.mp4` |
| `stream-rtsp` | RTSP camera stream | `stream-rtsp rtsp://user:pass@ip:554/stream` |
| `stream-webcam` | USB webcam | `stream-webcam [/dev/video0]` |
| `benchmark` | Benchmark (no output) | `benchmark` |
| `custom` | Custom command | `custom bash` |

## Performance

| Metric | Value |
|--------|-------|
| Model compile | ~2 minutes (first frame only) |
| Inference | ~5-10ms per frame |
| Throughput | 100-120 FPS after warmup |
| Resolution | 640x640 |

## Files

```
gstreamer_yolov8_demo/
├── Dockerfile          # Container build
├── entrypoint.sh       # Runtime modes
├── plugins/
│   └── yolov8s.py      # GStreamer plugin (TT inference)
├── mjpeg_server.py     # HTTP streaming server
└── README.md           # This file
```

## Requirements

- Tenstorrent Wormhole device
- Docker with TT device access
- Hugepages configured (`/dev/hugepages-1G`)
- Base image: `sdxl-inf-server_89c6a49:latest`

## Troubleshooting

**Plugin not found:**
```bash
docker run ... tt-gstreamer-yolov8 custom gst-inspect-1.0 yolov8s
```

**Model compilation stuck:**
- First run compiles trace (~2 min). Wait for "Model loaded" message.

**No output:**
- Check port mapping (`-p 8080:8080`)
- View container logs: `docker logs CONTAINER_ID`

## Architecture Options

| Option | Description | Use Case |
|--------|-------------|----------|
| **Option 1: Detection metadata** | Send JSON over WebSocket | Production CCTV (client has video) |
| **Option 2: Full video** | Stream annotated video | Demos, standalone viewers |
| **Option 3: HTTP inference** | tt-inference-server API | Batch processing, evals |

This demo implements **Option 2**. Option 1 (JSON metadata) is recommended for production CCTV deployments where the client already has the video stream.