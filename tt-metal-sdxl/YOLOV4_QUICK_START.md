# YOLOv4 Quick Start Guide

## Server is Running! 🎉

Your YOLOv4 inference server is now running on port **8050**.

### Access Points

- **API Documentation**: http://localhost:8050/docs
- **Object Detection Endpoint**: http://localhost:8050/cnn/search-image

### Test the API

#### Using Python:
```python
import requests
import base64

# Load and encode an image
with open("your_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8050/cnn/search-image",
    json={"prompt": image_base64},
    headers={"X-API-Key": "your-api-key"}
)

# Get detections
result = response.json()
if "image_data" in result:
    for detection in result["image_data"]:
        print(f"Detected: {detection['class_name']} "
              f"with confidence {detection['confidence']:.2f}")
```

#### Using cURL:
```bash
# First, encode your image to base64
base64 -w 0 image.jpg > image_base64.txt

# Send request
curl -X POST "http://localhost:8050/cnn/search-image" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d "{\"prompt\": \"$(cat image_base64.txt)\"}"
```

### Configuration Used

- **Model**: YOLOv4 (tt-yolov4)
- **Service**: CNN (Object Detection)
- **Port**: 8050
- **Classes**: 80 COCO classes
- **Resolution**: 320x320 (configurable)

### Stop the Server

```bash
# Find the process
ps aux | grep "python main.py"

# Kill it using the PID
kill <PID>
```

### Troubleshooting

If you encounter issues:

1. **Check if ttnn runtime is built**:
   ```bash
   cd /home/aroberge/tt-metal
   ./build_metal.sh
   ```

2. **Verify environment**:
   ```bash
   source /home/aroberge/tt-metal/python_env/bin/activate
   ```

3. **Check logs**:
   The server will output logs to the terminal showing any errors.

### Next Steps

- Try different images
- Adjust confidence thresholds in the runner
- Monitor performance metrics
- Integrate with your application

The server is ready to detect objects in your images!
