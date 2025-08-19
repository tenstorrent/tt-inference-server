# Supported Models

This inference server supports multiple models. Here's how to configure and use each one:

## Available Models

### 1. SDXL (Stable Diffusion XL)
- **Runner**: `tt-sdxl`
- **Service**: `image`
- **Endpoint**: `/image/generate`
- **Default model**

### 2. Stable Diffusion 3.5
- **Runner**: `tt-sd3.5`
- **Service**: `image`
- **Endpoint**: `/image/generate`

### 3. Whisper (Audio Transcription)
- **Runner**: `tt-whisper`
- **Service**: `audio`
- **Endpoint**: `/audio/transcribe`

### 4. YOLOv4 (Object Detection)
- **Runner**: `tt-yolov4`
- **Service**: `cnn`
- **Endpoint**: `/cnn/search-image`
- **Input**: Base64-encoded image
- **Output**: Object detections with bounding boxes and class names

## Configuration

To switch between models, update the environment variables:

```bash
# For YOLOv4
export MODEL_SERVICE=cnn
export MODEL_RUNNER=tt-yolov4
export NUM_INFERENCE_STEPS=1

# For SDXL (default)
export MODEL_SERVICE=image
export MODEL_RUNNER=tt-sdxl
export NUM_INFERENCE_STEPS=20
```

Or use the provided `.env` files:
- `.env.yolov4` - YOLOv4 configuration
- `.env.sdxl` - SDXL configuration (if available)

## Running the Server

1. Activate the tt-metal Python environment:
```bash
source /home/aroberge/tt-metal/python_env/bin/activate
```

2. Start the server:
```bash
cd /home/aroberge/tt-inference-server/tt-metal-sdxl
python main.py
```

## Requirements

All models require:
- Properly built tt-metal environment
- TT hardware device
- `python_env` virtual environment from tt-metal
