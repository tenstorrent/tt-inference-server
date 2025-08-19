# YOLOv4 Setup Guide

## Quick Start

1. **Activate tt-metal environment**:
```bash
source /home/aroberge/tt-metal/python_env/bin/activate
```

2. **Configure for YOLOv4**:
```bash
export MODEL_SERVICE=cnn
export MODEL_RUNNER=tt-yolov4
```

3. **Run the server**:
```bash
cd /home/aroberge/tt-inference-server/tt-metal-sdxl
python main.py
```

## API Usage

Send a POST request to `/cnn/search-image`:

```python
import requests
import base64

# Encode image
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/cnn/search-image",
    json={"prompt": image_base64},
    headers={"X-API-Key": "your-api-key"}
)

# Get detections
detections = response.json()["image_data"]
```

## Files Added for YOLOv4

- `tt_model_runners/yolov4_runner.py` - Main runner implementation
- `tt_model_runners/resources/coco.names` - COCO class names
- `tests/test_yolov4_runner.py` - Unit tests
- Configuration files and documentation

## Requirements

- tt-metal built and installed
- TT hardware device
- Dependencies: opencv-python, pillow (added to requirements.txt)
