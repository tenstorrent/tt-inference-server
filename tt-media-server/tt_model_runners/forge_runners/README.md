
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
   cd tt_model_runners
   pip install -r requirements.txt
   ```

## Usage

### Starting the Server

Launch the inference server on port 8000 (from tt-media-server folder):
```bash
uvicorn main:app --lifespan on --port 8000
```

The server will be available at: `http://127.0.0.1:8000`

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://127.0.0.1:8000/docs`

### Making Inference Requests

#### Using cURL
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/image/generations' \
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
