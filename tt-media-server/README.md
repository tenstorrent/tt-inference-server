# TT non-LLM inference server

This server is built to serve non-LLM models. Currently supported models:

1. SDXL-trace
2. SD3.5
3. Whisper
4. Microsoft Resnet (Forge)

# Repo structure

1. Config - config files that can be overridden by environment variables.
2. Domain - Domain and transfer objects
3. Model services - Services for processing models, scheduler for models and a runner
4. Open_ai_api - controllers in OpenAI flavor
5. Resolver - creator of scheduler and model, depending on the config creates singleton instances of scheduler and model service
6. Security - Auth features
7. Tests - general end to end tests
8. Model runners - runners for devices and models. Runner_fabric is responsible for creating a needed runner

More details about each folder will be provided below

# Installation instructions

To just run a server build a docker file and run it.

For development running:

1. Setup tt-metal and all the needed variables for it
2. Make sure you're in tt-metal's python env
3. Clone tt-inference-server repo and switch to dev branch
4. ```sudo apt update && sudo apt install -y ffmpeg && pip install -r requirements.txt``` from tt-media-server
5. ```uvicorn main:app --lifespan on --port 8000``` (lifespan methods are needed to init device and close the devices)

## SDXL setup

### Standard SDXL Setup
1. ```export MODEL_RUNNER=tt-sdxl-trace```
2. Run the server ```uvicorn main:app --lifespan on --port 8000```

### SDXL with Tensor Parallelism (TP2)
1. ```export TP2=true```
2. ```export MODEL_RUNNER=tt-sdxl-trace```
3. Run the server ```source run_uvicorn.sh```

**Note:** TP2 configuration requires exactly 2 TT devices and is only supported for SDXL models.

## SD-3.5 setup

Its easiest to use the [Special Environment Variable Overrides](#special-environment-variable-overrides) to help create the necessary setup for the target device.

### Standard SD-3.5 Setup
1. Set the model special env variable ```export MODEL=stable-diffusion-3.5-large```
2. Set device special env variable ```export DEVICE=galaxy``` or ```export DEVICE=t3k```
3. Run the server ```uvicorn main:app --lifespan on --port 8000```

### SD-3.5 with Custom Device Mesh Configurations

For optimized performance, you can use pre-configured device mesh setups:

#### Base Configuration (8 devices: 2x4 mesh)
```bash
export SD_3_5_BASE=true
export MODEL=stable-diffusion-3.5-large
export DEVICE=galaxy
source run_uvicorn.sh
```

#### Fast Configuration (32 devices: 4x8 mesh)
```bash
export SD_3_5_FAST=true
export MODEL=stable-diffusion-3.5-large
export DEVICE=galaxy
source run_uvicorn.sh
```

**Important Notes:**
- Base configuration requires 8 TT devices arranged in a 2x4 mesh
- Fast configuration requires 32 TT devices arranged in a 4x8 mesh
- Only Galaxy hardware with sufficient devices is supported
- Choose the configuration based on your hardware availability and performance requirements

Please note that only quietbox and 6u galaxy are supported.

## Audio Preprocessing Setup and Model Terms

When setting `allow_audio_preprocessing` for the first time and testing audio models, you must:

**Accept Terms for All Required Models:**
1. Main diarization model: https://hf.co/pyannote/speaker-diarization-3.0
2. Segmentation model: https://hf.co/pyannote/segmentation-3.0

- For Company/University, enter: `Tenstorrent Inc.`
- For Website, enter: `https://tenstorrent.com`

**Hugging Face Token Setup:**
- Create a Hugging Face token on the HF website with read permission.
- Export the token as an environment variable:

```bash
export HF_TOKEN=[copied token]
```

This is required for downloading and using the models during audio preprocessing.


## Testing instructions

If server is running in development mode (ENVIRONMENT=development), OpenAPI endpoint is available on /docs URL.

# Image generation test call

Sample for calling the endpoint for image generation via curl:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/image/generations' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Volcano on a beach",
  "negative_prompt": "low quality",
  "num_inference_steps": 20,
  "seed": 0,
  "guidance_scale": 7.0,
  "number_of_images": 1
}'
```

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable.

# Audio transcription test call

The audio transcription API supports multiple audio formats and input methods with automatic format detection and conversion.

- Base64 JSON Request: Send a JSON POST request to `/audio/transcriptions`
Sample for calling the audio transcription endpoint via curl:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/audio/transcriptions' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  --data-binary @server/tests/test_data.json \
  --no-buffer
```

test_data.json file example:
```bash
{
    "stream": false,
    "file": "[base64 audio file]"
}
```

- File Upload (WAV/MP3): Send a multipart form data POST request to `/audio/transcriptions`
```bash
# WAV file upload
curl -X POST "http://localhost:8000/audio/transcriptions" \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@/path/to/audio.wav" \
  -F "stream=true" \
  -F "is_preprocessing_enabled=true" \
  --no-buffer
```

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable.

*Please note that test_data.json is within docker container or within tests folder*

# Configuration

The TT Inference Server can be configured using environment variables or by modifying the settings file. All parameter names should be **UPPERCASED** when used as environment variables.

## General Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `MODEL_RUNNER` | [`ModelRunner.TT_SDXL_TRACE.value`](config/constants.py ) | Specifies the type of runner to run |
| `LOG_LEVEL` | `"INFO"` | Sets the logging level for the application. Valid values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `ENVIRONMENT` | `"development"` | Specifies the runtime environment. Used for environment-specific configurations |
| `LOG_FILE` | `None` | Optional path to log file. If not set, logs are output to console only |
| `MODEL` | `None` | Specifies the model to run. Used in combination with DEVICE for config override |
| `DEVICE` | `None` | Specifies the target device type for model execution. Used in combination with MODEL for config override |
| `SD_3_5_FAST` | `None` | Configures device mesh for SD-3.5 in fast configuration (4x8 mesh = 32 devices) |
| `SD_3_5_BASE` | `None` | Configures device mesh for SD-3.5 in base configuration (2x4 mesh = 8 devices) |
| `TP2` | `None` | Enables tensor parallelism across 2 devices. Compatible with SDXL models only |
| `HF_TOKEN` | `None` | Hugging Face token with read permission for accessing models during audio preprocessing |

## Device Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `DEVICE_IDS` | `"(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15),(16),(17),(18),(19),(20),(21),(22),(23),(24),(25),(26),(27),(28),(29),(30),(31)"` | Comma-separated list of device IDs available for inference. Defines which TT devices can be used |
| `IS_GALAXY` | `True` | Boolean flag indicating if running on Galaxy hardware. Used for graph device split and class initialization |
| `DEVICE_MESH_SHAPE` | `(1, 1)` | Tuple defining the device mesh topology. Format: `(rows, columns)` for multi-device setups |
| `RESET_DEVICE_COMMAND` | `"tt-smi -r"` | Command used to reset TT devices when needed |
| `RESET_DEVICE_SLEEP_TIME` | `5.0` | Time in seconds to wait after device reset before attempting reconnection |

## Model Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `MODEL_RUNNER` | [`ModelRunners.TT_SDXL_TRACE.value`](config/constants.py ) | Specifies which model runner implementation to use for inference |
| `MODEL_WEIGHTS_PATH` | `""` | Path or HuggingFace model ID for the model weights to load |
| `TRACE_REGION_SIZE` | `34541598` | Memory size allocated for model tracing operations (in bytes) |

## Queue and Batch Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `MAX_QUEUE_SIZE` | `64` | Maximum number of requests that can be queued for processing |
| `MAX_BATCH_SIZE` | `1` | Maximum batch size for inference requests. Currently limited to 1 for stability |

## Worker Management

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `NEW_DEVICE_DELAY_SECONDS` | `30` | Delay in seconds before initializing a new device worker |
| `MOCK_DEVICES_COUNT` | `5` | Number of mock devices to create when running in mock/test mode |
| `MAX_WORKER_RESTART_COUNT` | `5` | Maximum number of times a worker can be restarted before being marked as failed |
| `WORKER_CHECK_SLEEP_TIMEOUT` | `30.0` | Time in seconds between worker health checks |

## Timeout Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `DEFAULT_INFERENCE_TIMEOUT_SECONDS` | `60` | Default timeout for inference requests in seconds (1 minute) |

## Image Generation Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `NUM_INFERENCE_STEPS` | `20` | Number of denoising steps for image generation. Currently hardcoded and cannot be overridden per request |

## Audio Processing Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `MAX_AUDIO_DURATION_SECONDS` | `60.0` | Maximum allowed audio duration for transcription requests (in seconds) |
| `MAX_AUDIO_SIZE_BYTES` | `52428800` | Maximum allowed audio file size (50 MB in bytes) |
| `DEFAULT_SAMPLE_RATE` | `16000` | Default audio sample rate for processing (16 kHz) |
| `ALLOW_AUDIO_PREPROCESSING` | `True` | Boolean flag to allow audio preprocessing capabilities |

## Authentication Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `API_KEY` | `"your-secret-key"` | Secret key used for API authentication. All requests must include `Authorization: Bearer <API_KEY>` header |

## Special Environment Variable Overrides

The server supports special environment variable combinations that can override multiple settings at once:

| Environment Variable | Description |
|---------------------|-------------|
| `MODEL` | Combined with `DEVICE`, overrides configuration based on predefined ModelConfigs |
| `DEVICE` | Combined with `MODEL`, overrides configuration based on predefined ModelConfigs |

When both `MODEL` and `DEVICE` are set, the server will look up the corresponding configuration in [`ModelConfigs`](config/constants.py ) and apply all associated settings automatically.

## Device Mesh Configuration

The server supports special environment variables for configuring device mesh shapes for specific model configurations:

| Environment Variable | Device Mesh Shape | Description |
|---------------------|-------------------|-------------|
| `TP2` | `(2, 1)` | Enables tensor parallelism across 2 devices. **Compatible with SDXL models only** |
| `SD_3_5_BASE` | `(2, 4)` | Configures device mesh for Stable Diffusion 3.5 in base configuration (8 devices total) |
| `SD_3_5_FAST` | `(4, 8)` | Configures device mesh for Stable Diffusion 3.5 in fast configuration (32 devices total) |

### Usage Examples

#### Running SDXL with Tensor Parallelism (TP2)
```bash
# Enable TP2 for SDXL (requires 2 devices)
export TP2=true
export MODEL_RUNNER=tt-sdxl-trace
source run_uvicorn.sh
```

**Note:** TP2 configuration is currently supported only for SDXL models and requires exactly 2 TT devices.

#### Running Stable Diffusion 3.5 Base Configuration
```bash
# SD-3.5 base setup (2x4 mesh = 8 devices)
export SD_3_5_BASE=true
export MODEL=stable-diffusion-3.5-large
export DEVICE=galaxy
source run_uvicorn.sh
```

#### Running Stable Diffusion 3.5 Fast Configuration
```bash
# SD-3.5 fast setup (4x8 mesh = 32 devices)
export SD_3_5_FAST=true
export MODEL=stable-diffusion-3.5-large
export DEVICE=galaxy
source run_uvicorn.sh
```

**Important Notes:**
- These environment variables override the default `DEVICE_MESH_SHAPE` setting
- SD-3.5 configurations require Galaxy hardware with sufficient devices or T3K

## Configuration File

The server also supports configuration via a `.env` file in the project root. Environment variables take precedence over `.env` file settings.

## Configuration Examples

### Basic Configuration
```bash
# Set log level to debug
export LOG_LEVEL=DEBUG

# Configure for specific devices only
# Brackets represent chip pairs that will be grouped together
export DEVICE_IDS="(0,1),(2,3)"
```

### High-Throughput Configuration
```bash
# Increase queue size for high-throughput scenarios
export MAX_QUEUE_SIZE=128

# Set custom timeout for long-running inferences
export DEFAULT_INFERENCE_TIMEOUT_SECONDS=300
```

### Production Configuration
```bash
# Configure for production environment
export ENVIRONMENT=production
export LOG_FILE="/var/log/tt-inference-server.log"
export LOG_LEVEL=WARNING
```

### Model and Device Override
```bash
# Use predefined model/device configuration
export MODEL="stable-diffusion-xl-base-1.0"
export DEVICE="n300"
```

### Audio Processing Configuration
```bash
# Configure for longer audio files
export MAX_AUDIO_DURATION_SECONDS=300.0
export MAX_AUDIO_SIZE_BYTES=104857600  # 100 MB
export DEFAULT_SAMPLE_RATE=22050
export ALLOW_AUDIO_PREPROCESSING=true
```

### Authentication Configuration
```bash
# Set custom API key for authentication
export API_KEY="my-secure-secret-key-123"

# For production, use a strong random key
export API_KEY="$(openssl rand -base64 32)"
```

When `API_KEY` is set, all API requests must include the authorization header:
```bash
# Example with custom API key
curl -H "Authorization: Bearer my-secure-secret-key-123" \
     ...
```

### Development Configuration
```bash
# Use mock devices for development
export MOCK_DEVICES_COUNT=2
export DEVICE_IDS="(0),(1)"
export ENVIRONMENT=development
```


# Steps for Onboarding a Model to the Inference Server

If you're integrating a new model into the inference server, here’s a suggested workflow to help guide the process:

1. **Implement a Model Runner** Create a model runner by inheriting the *base_runner* class and implementing its abstract methods. You can find the relevant codebase here: [tt-inference-server/tt-media-server/tt_model_runners at dev · tenstorrent/tt-inference-server ](https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-media-server/tt_model_runners)
(most likely a model runner is a *demo.py* file from a model in tt-metal broken down in methods of a class)
2. **Update Dependencies** If your runner relies on any additional libraries, please make sure to add them to the requirements.txt:  [tt-inference-server/tt-media-server/requirements.txt at dev · tenstorrent/tt-inference-server ](https://github.com/tenstorrent/tt-inference-server/blob/dev/tt-media-server/requirements.txt)
3. **Modify *runner_fabric.py*** Update *runner_fabric.py* to instantiate your runner based on the configuration: [tt-inference-server/tt-media-server/tt_model_runners/runner_fabric.py at dev · tenstorrent/tt-inference-server ](https://github.com/tenstorrent/tt-inference-server/blob/dev/tt-media-server/tt_model_runners/runner_fabric.py)
4. **Add a Dummy Config** Add a basic config entry to help instantiate your runner: [tt-inference-server/tt-media-server/config/settings.py at dev · tenstorrent/tt-inference-server ](https://github.com/tenstorrent/tt-inference-server/blob/dev/tt-media-server/config/settings.py)
Alternatively, you can use an environment variable:
```export MODEL_RUNNER=<your-model-runner-name>```
5. **Write a Unit Test** Please include a unit test in the *tests/* folder to verify your runner works as expected. This step is crucial—without it, it’s difficult to pinpoint issues if something breaks later
6. **Open an Issue for CI Coverage** Kindly submit a GitHub issue for Igor Djuric to review your PR and to help cover end to end running, CI integration, or any missing service steps: [https://github.com/tenstorrent/tt-inference-server/issuesConnect your Github account ](https://github.com/tenstorrent/tt-inference-server/issues)
7. **Share Benchmarks (if available)** If you’ve run any benchmarks or evaluation tests, please share them. They’re very helpful for understanding performance and validating correctness.

# Docker build and run

Docker build sample:

```bash
docker build -t sdxl-inf-server --platform=linux/amd64 -f tt-media-server/Dockerfile .
```

Docker image link:

https://github.com/tenstorrent/tt-inference-server/pkgs/container/tt-inference-server%2Ftt-server-dev-ubuntu-22.04-amd64

Docker run sample:

```bash
docker run \
  -e MODEL_RUNNER=forge \
  --rm -it \
  -p 8000:8000 \
  --user root \
  --entrypoint "/bin/bash" \
  --device /dev/tenstorrent/0 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  ghcr.io/tenstorrent/tt-inference-server/tt-server-dev-ubuntu-22.04-amd64
```

**Suggestion:** Always take the latest docker image

## Galaxy running settings

Running SDXL on Galaxy:

```bash
sudo docker run -d -it \
  -e MODEL_RUNNER=tt-sdxl-trace \
  -e DEVICE_IDS="(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15),(16),(17),(18),(19),(20),(21),(22),(23)" \
  --cap-add=sys_nice \
  --security-opt seccomp=unconfined \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --device /dev/tenstorrent \
  -p 8000:8000 \
  --user root \
  --device /dev/ipmi0 \
  ghcr.io/tenstorrent/tt-inference-server/tt-server-dev-ubuntu-22.04-amd64
```

**Note:** Sample above will run 24 devices with numbers 0 to 23. Please note it'd be a good practice to mount only the devices you are planning to use to avoid collisions.

Running Whisper on Galaxy:

```bash
sudo docker run -d -it \
  -e MODEL_RUNNER=tt-whisper \
  -e DEVICE_IDS="(24),(25),(26)" \
  --cap-add=sys_nice \
  --security-opt seccomp=unconfined \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --device /dev/tenstorrent \
  -p 8000:8000 \
  --user root \
  --device /dev/ipmi0 \
  ghcr.io/tenstorrent/tt-inference-server/tt-server-dev-ubuntu-22.04-amd64
```

**Note:** Sample above will run Whisper model on devices 24 to 26 - 3 devices.

# Remaining work:

1. Add unit tests
2. Add API tests
3. Cleanup unused things in runners
