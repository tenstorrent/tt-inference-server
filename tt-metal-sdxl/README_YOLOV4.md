# YOLOv4 Object Detection Service

**Note**: This service requires a properly built tt-metal environment with the `python_env` virtual environment activated.

This implementation provides a YOLOv4 object detection service using Tenstorrent hardware acceleration, integrated into the tt-inference-server framework.

## Architecture Overview

The YOLOv4 service follows the established pattern of the tt-inference-server:

1. **API Layer** - FastAPI endpoints for object detection requests
2. **Service Layer** - CNN service handling request processing and queuing
3. **Runner Layer** - TTYolov4Runner managing model execution on TT hardware
4. **Worker Processes** - Device workers processing inference requests

## Features

- **Real-time object detection** using YOLOv4 on Tenstorrent hardware
- **80 COCO classes** supported out of the box
- **Configurable confidence and NMS thresholds**
- **Base64 image input/output** for easy API integration
- **Batch processing support** for multiple images
- **Automatic device management** with mesh configuration

## Configuration

### Environment Variables

Copy `.env.yolov4.example` to `.env` and configure:

```bash
MODEL_SERVICE=cnn
MODEL_RUNNER=tt-yolov4
DEVICE_IDS=0
MAX_BATCH_SIZE=1
```

### Device Requirements

- Wormhole B0 device recommended
- L1 small size: 10960
- Trace region size: 6434816
- Command queues: 2

## API Usage

### Object Detection Endpoint

**POST** `/cnn/search-image`

Request body:
```json
{
    "prompt": "<base64-encoded-image>"
}
```

Response:
```json
{
    "image_data": [
        {
            "bbox": {
                "x1": 0.1,
                "y1": 0.2,
                "x2": 0.3,
                "y2": 0.4
            },
            "confidence": 0.95,
            "class_id": 0,
            "class_name": "person"
        }
    ],
    "status": "success"
}
```

### Supported Image Formats

- Input: Base64-encoded PNG/JPEG images
- Resolutions: 320x320, 640x640 (configurable)
- Color mode: RGB

## Model Details

### YOLOv4 Architecture

The implementation uses the performant YOLOv4 runner from tt-metal with:
- Multi-scale detection (3 YOLO layers)
- Anchor-based bounding box prediction
- 80 COCO object classes
- Optimized TTNN operations with tracing

### Post-Processing

- Non-Maximum Suppression (NMS) with configurable threshold
- Confidence filtering
- Bounding box coordinate normalization
- Class name mapping

## Running the Service

### Start the Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the service
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Test with Sample Image

```python
import base64
import requests
from PIL import Image
from io import BytesIO

# Load and encode image
with open("test_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/cnn/search-image",
    json={"prompt": image_base64},
    headers={"X-API-Key": "your-api-key"}
)

# Process results
detections = response.json()["image_data"]
for det in detections:
    print(f"Detected {det['class_name']} with confidence {det['confidence']:.2f}")
```

## Performance Optimization

### Batching
- Configure `MAX_BATCH_SIZE` for batch processing
- Workers automatically batch requests from queue

### Tracing
- Model uses TTNN trace capture for optimized inference
- Two command queues for parallel host-device transfers

### Memory Configuration
- DRAM sharded input for efficient data transfer
- Optimized L1 memory allocation for convolutions

## Troubleshooting

### Common Issues

1. **Import Error for YOLOv4PerformantRunner**
   - Ensure tt-metal is properly installed
   - Check PYTHONPATH includes tt-metal directory

2. **Device Not Found**
   - Verify device IDs with `tt-smi`
   - Check device permissions

3. **Out of Memory**
   - Reduce batch size
   - Use smaller resolution (320x320)

### Logging

Enable debug logging:
```bash
LOG_LEVEL=DEBUG uvicorn main:app
```

## Development

### Running Tests

```bash
# Run YOLOv4 specific tests
pytest tests/test_yolov4_runner.py -v

# Run all tests
pytest tests/ -v
```

### Adding Custom Classes

To use custom classes instead of COCO:

1. Create custom names file:
```bash
echo "class1\nclass2\nclass3" > tt_model_runners/resources/custom.names
```

2. Modify runner to load custom names:
```python
self.class_names = self._load_class_names("custom.names")
```

## Architecture Details

### Service Flow

```
User Request → API Layer → CNN Service → Task Queue
                                            ↓
                                      Device Worker
                                            ↓
                                    TTYolov4Runner
                                            ↓
                                    YOLOv4PerformantRunner
                                            ↓
                                      TT Hardware
                                            ↓
                                     Post-Processing
                                            ↓
                                     Response Queue
                                            ↓
                                        API Response
```

### Key Components

- **TTYolov4Runner**: Main runner class handling model loading and inference
- **YOLOv4PerformantRunner**: Optimized TTNN implementation with tracing
- **CNNService**: Service layer for request processing
- **DeviceWorker**: Process managing hardware device and inference loop

## Future Improvements

- [ ] Support for custom model weights
- [ ] Dynamic resolution selection
- [ ] Video stream processing
- [ ] Multi-device parallelism
- [ ] TensorRT/ONNX export support
- [ ] Real-time visualization endpoint
