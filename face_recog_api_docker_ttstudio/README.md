# Face recognition API (FastAPI + YuNet + SFace)

Runs on **`ghcr.io/tenstorrent/tt-inference-server/tt-metal-sface-yolov8s:blackhole`** (or compatible).

## Build

```bash
cd ~/teja/face-recognition-api
docker build -t face-recognition-api:local .
```

## Run

```bash
docker run --rm -it -p 7000:7000 \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  -v "$(pwd)/registered_faces:/app/registered_faces" \
  -e DEVICE_ID=0 \
  -e RECOGNITION_THRESHOLD=0.5 \
  face-recognition-api:local
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Readiness + model load state |
| POST | `/register-face` | `multipart`: `name`, `image` |
| POST | `/recognize-face` | `multipart`: `image` |
| GET | `/registered-faces` | List names |
| DELETE | `/registered-faces/{name}` | Remove identity |

## Quick test

```bash
curl -s http://localhost:7000/health
curl -s -X POST http://localhost:7000/register-face -F "name=demo" -F "image=@photo.jpg"
curl -s -X POST http://localhost:7000/recognize-face -F "image=@photo.jpg"
```
