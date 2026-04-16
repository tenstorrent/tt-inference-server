
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
   - TT_XLA_EFFICIENTNET = "tt-xla-efficientnet"
   - TT_XLA_SEGFORMER = "tt-xla-segformer"
   - TT_XLA_UNET = "tt-xla-unet"
   - TT_XLA_VIT = "tt-xla-vit"
   - TT_XLA_SDXL = "tt-xla-sdxl"

Set appropriate HF_TOKEN to load weights from Huggingface.
IRD_LF_CACHE is out large file caching service, in IRD environment use http://aus2-lfcache.aus2.tenstorrent.com/

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
  'http://127.0.0.1:8000/v1/cnn/search-image' \
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

## Onboard a New Forge LLM Model to TT-Inference-Server

### Repository Changes

#### tt-media-server
- Add the new model to `ModelNames` and `SupportedModels` enums if they don't already exist there

#### tt-inference-server
- Add new model spec with forge vllm plugin implementation (model_spec.py)
- If the model does not exist in eval_config.py, you can add your eval_config, but the eval tasks should be the same as other tasks from other eval_configs from the model's family. Check NOTES below as well.

#### tt-shield
- Add model names in `on-dispatch.yml` dropdown when selecting models

### Local Testing

Model should first be tested locally:

1. In tt-media-server, in `config/vllm_settings`, choose the desired model from the `SupportedModels` enum
2. In `config/settings.py`, set your device id(s), `is_galaxy` bool, and most importantly, `model_runner` to `ModelRunners.VLLM.value`
3. Create a python3.12 venv with the forge vllm plugin and activate
4. Do a `pip install -r` in both tt-inference-server and tt-media-server
5. Do `export VLLM_TARGET_DEVICE="empty"`
6. Run the tt-media-server with python3.12 venv (exec the `run_uvicorn.sh`)
7. You can send completion requests via `localhost:8000/docs` page

### CI

To run the forge model, select the `forge-vllm-plugin` implementation when running the dispatch workflow in tt-shield. This will trigger the building of a tt-media-server container running the forge vllm plugin.

Add the model into the options dropdown(under the model input) in on-dispatch.yml in .github/workflows/on-dispatch.yml file

To add models into the on-nightly workflow, navigate to tt-shield repo, and add the model into the model matrix in .github/workflows/on-dispatch.yml , under   run-evals-on-media-inference-server-forge job

### Model Specific Options

- For TT_XLA_SDXL use TTXLA_SDXL_RESOLUTION [512|1024] to specify the output image resolution, default is 512.

NOTES:
- We are unable to run evaluations on Forge models that exist in metal.
Reasons:
   - current eval_configs send a seed parameter, which xla does not support per request
   - current eval_configs specify max_model_lenght that will crash the FORGE LLM model.
   - We cannot support two eval_configs for the same model at the moment.
   - If the model only exists on forge, eval can be ran, but make sure that you limit max model len + add the limit of how many requests eval will send(limit_samples_map)
