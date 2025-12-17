
# Forge Runner Module

The Forge Runner is a device runner implementation that uses TT Forge for model compilation and inference on Tenstorrent hardware.

## Overview

This module provides:
- Model loading and compilation using TT Forge
- Inference execution on Tenstorrent devices
- Integration with the inference server

### Installation

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv-worker
   source venv-worker/bin/activate
   ```

2. **Navigate to project directory:**
   ```bash
   cd tt-media-server/
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install forge dependencies:**
   ```bash
   pip install -r tt_model_runners/forge_runners/requirements.txt
   ```

## Usage

### Starting the Server

Set the model runner and launch the inference server on port 8000 (from tt-media-server folder).

- Device ID is the id of tenstorrent device you are using
```ls /dev/tenstorrent```

Use MODEL_RUNNER to select which model is run
   - TT_XLA_RESNET = "tt-xla-resnet"
   - TT_XLA_VOVNET = "tt-xla-vovnet"
   - TT_XLA_MOBILENETV2 = "tt-xla-mobilenetv2"
   - TT_XLA_EFFICENNET = "tt-xla-efficientnet"
   - TT_XLA_SEGFORMER = "tt-xla-segformer"
   - TT_XLA_UNET = "tt-xla-unet"
   - TT_XLA_VIT = "tt-xla-vit"

Set appropriate HF_TOKEN to load weights from Huggingface.
IRD_LF_CACHE is out large file caching service, in IRD enviroment use http://aus2-lfcache.aus2.tenstorrent.com/

```bash
export MODEL_RUNNER=tt-xla-resnet
export DEVICE_IDS="3"
export HF_TOKEN=<HF Token>
export IRD_LF_CACHE=http://aus2-lfcache.aus2.tenstorrent.com/
uvicorn main:app --lifespan on --port 8000
```

The server will be available at: `http://127.0.0.1:8000`

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://127.0.0.1:8000/docs`

### Demo

Resnet Demo 
- http://127.0.0.1:8000/static/demos/resnet.html

### Making Inference Requests

#### Using cURL
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/cnn/search-image' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A beautiful landscape painting"
  }'
```

## Development

### Running Tests

Execute the test suite:
```bash
pytest tests/test_forge_runner.py -v
```


```
pip install pytest
pip install pytest-asyncio
tt-inference-server/tt-media-server$ pytest tt_model_runners/forge_runners/test_forge_models.py
```