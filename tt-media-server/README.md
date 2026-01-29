# TT non-LLM inference server

This server is built to serve non-LLM models. Currently supported models:

1. SDXL-trace
2. SDXL-image-to-image
3. SDXL-edit
4. SD3.5
5. Flux1
6. Mochi1
7. Wan2.2
8. Motif-Image-6B-Preview
9. Qwen-Image
10. Whisper
11. Microsoft Resnet (Forge)
12. VLLM with TT Plugin

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

### SDXL Image To Image Setup
1. ```export MODEL_RUNNER=tt-sdxl-image-to-image```
2. Run the server ```uvicorn main:app --lifespan on --port 8000```

### SDXL Edit Setup
1. ```export MODEL_RUNNER=tt-sdxl-edit```
2. Run the server ```uvicorn main:app --lifespan on --port 8000```


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
- Only Galaxy and T3K hardware with sufficient devices is supported
- Choose the configuration based on your hardware availability and performance requirements


## Supported DiT models
The setup for other supported DiT models is very similar to [Standard SD-3.5 Setup](#standard-sd-35-setup). Choose a configuration from the table below, and run the server.

| MODEL | Supported device options|
|-------|--------|
| stable-diffusion-3.5-large | galaxy, t3k |
| flux.1-dev | galaxy, t3k, p300, qbge |
| flux.1-schnell | galaxy, t3k, p300, qbge |
| motif-image-6b-preview | galaxy, t3k |
| qwen-image | galaxy, t3k |
| qwen-image-2512 | galaxy, t3k |
| mochi-1-preview | galaxy, t3k |
| Wan2.2-T2V-A14B-Diffusers | galaxy, t3k, qbge |

For example, to run flux.1-dev on t3k
1. Set the model special env variable e.g ```export MODEL=flux.1-dev```.
2. Set device special env variable e.g ```export DEVICE=t3k```.
3. Run the server ```uvicorn main:app --lifespan on --port 8000```.

## VLLM with TT Plugin Setup

The server supports running large language models using VLLM with the Tenstorrent plugin.

### Prerequisites

1. **Install the TT-VLLM Plugin**

   Follow the installation instructions from the repository:
   https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-vllm-plugin

2. **Required Environment Variables**

   ```bash
   # Specify the Hugging Face model to use
   export HF_MODEL='meta-llama/Llama-3.1-8B-Instruct'

   # Enable VLLM V1 API
   export VLLM_USE_V1=1

   # Set the model runner
   export MODEL_RUNNER=vllm-forge
   ```

3. **Run the Server**

### Testing VLLM Completions

Once the server is running, you can test text completion using curl. The VLLM endpoint supports streaming responses by default. Tokens will be returned as they are generated:


```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/v1/completions' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Write a short story about a robot",
    "max_tokens": 500,
    "temperature": 0.8
  }' \
  --no-buffer
```

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable.

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

# Audio transcription and translation test call

The audio transcription and translation API supports multiple audio formats and input methods with automatic format detection and conversion.

- Base64 JSON Request: Send a JSON POST request to `/audio/transcriptions` or `/audio/translations`
Sample for calling the audio transcription/translations endpoint via curl:
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

- File Upload (WAV/MP3): Send a multipart form data POST request to `/audio/transcriptions` or `/audio/translations`
```bash
# WAV file upload
curl -X POST "http://localhost:8000/audio/transcriptions" \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@/path/to/audio.wav" \
  -F "stream=true" \
  -F "is_preprocessing_enabled=true" \
  -F "perform_diarization=false" \
  -F "temperatures=0.0,0.2,0.4,0.6,0.8,1.0" \
  -F "compression_ratio_threshold=2.4" \
  -F "logprob_threshold=-1.0" \
  -F "no_speech_threshold=0.6" \
  -F "return_timestamps=true" \
  -F "prompt=test" \
  --no-buffer
```

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable.

*Please note that test_data.json is within docker container or within tests folder*


# Text-to-Speech (TTS) test call

The Text-to-Speech API converts text to speech audio using the SpeechT5 model.

- JSON Request: Send a JSON POST request to `/audio/speech`

**Default behavior:** Returns WAV file directly (default `response_format="audio"`)

```bash
curl -X POST 'http://127.0.0.1:8000/audio/speech' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hello, this is a test of the text to speech system."
  }' \
  --output output.wav \
  --silent \
  --show-error
```

**Request WAV file with explicit format:**

```bash
curl -X POST 'http://127.0.0.1:8000/audio/speech' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hello world, this is a test of text to speech",
    "response_format": "audio"
  }' \
  --output output.wav \
  --silent \
  --show-error
```

**Request JSON response with base64 audio:**

```bash
curl -X POST 'http://127.0.0.1:8000/audio/speech' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "This should return JSON",
    "response_format": "verbose_json"
  }' \
  --silent \
  --show-error
```

**Swagger/OpenAPI request body examples:**

```json
{
  "text": "Hello, this is a test of the text to speech system."
}
```

```json
{
  "text": "Hello world, this is a test of text to speech",
  "response_format": "audio"
}
```

```json
{
  "text": "This is another test",
  "response_format": "wav"
}
```

```json
{
  "text": "This should return JSON",
  "response_format": "verbose_json"
}
```

```json
{
  "text": "This is a JSON format test",
  "response_format": "json"
}
```

```json
{
  "text": "Hello, this is a test of the text to speech system.",
  "response_format": "audio",
  "speaker_id": "default_speaker"
}
```

**Available response formats:**
- `"audio"` or `"wav"` (default) - Returns WAV file directly (binary, `Content-Type: audio/wav`)
- `"verbose_json"` or `"json"` - Returns JSON with base64-encoded audio

**Optional fields:**
- `speaker_id` - ID for pre-configured speaker embeddings (0-7456 for CMU ARCTIC dataset)
- `speaker_embedding` - Base64-encoded or raw bytes of speaker embedding (advanced)

**Note:** Do NOT include `speaker_embedding` unless you have a valid base64-encoded embedding.

# Image search test call

The image search API uses a CNN model to search for similar images. It supports multiple input methods.

- Base64 JSON Request: Send a JSON POST request to `/search-image`
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/search-image' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "[base64 encoded image]",
  "response_format": "json",
  "top_k": 3,
  "min_confidence": 70.0
}'
```

- File Upload: Send a multipart form data POST request to `/search-image`
```bash
curl -X POST "http://localhost:8000/search-image" \
  -H "Authorization: Bearer your-secret-key" \
  -F "file=@/path/to/image.jpg" \
  -F "response_format=json" \
  -F "top_k=5" \
  -F "min_confidence=80.0"
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` / `file` | string / file | required | Base64-encoded image (JSON) or image file (multipart) |
| `response_format` | string | `"json"` | Response format for results |
| `top_k` | integer | `3` | Number of top results to return |
| `min_confidence` | float | `70.0` | Minimum confidence threshold (0-100) |

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable.

# Video generation API

## Submit video generation job

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/video/generations' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Volcano on a beach",
  "negative_prompt": "low quality",
  "num_inference_steps": 20
}'
```

**Response example:**
```json
{
  "id": "video_id_1",
  "object": "video",
  "status": "queued",
  "created_at": 1702860000,
  "model": "Wan2.2-T2V-A14B-Diffusers"
}
```

Save the `id` field from the response (e.g., `video_id_1`) to use as `{video_id}` in subsequent requests.

## Get video job metadata

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/video/generations/{video_id}' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key'
```

## Download generated video

The `/video/generations/{video_id}/download` endpoint for downloading a video file

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/video/generations/{video_id}/download' \
  -H 'Authorization: Bearer your-secret-key' \
  -o output.mp4
```

## Cancel video job and assets

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/video/generations/{video_id}/cancel' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key'
```

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable.

# Fine-tuning API

## Create fine-tuning job

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/fine_tuning/jobs' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "training_file": "file-abc123",
  "hyperparameters": {
    "n_epochs": 3,
    "batch_size": 4,
    "learning_rate_multiplier": 1.0
  }
}'
```

**Response example:**
```json
{
  "id": "ftjob-abc123",
  "object": "training",
  "status": "queued",
  "created_at": 1702860000,
  "model": "meta-llama/Llama-3.1-8B-Instruct"
}
```

Save the `id` field from the response (e.g., `ftjob-abc123`) to use as `{job_id}` in subsequent requests.

## List fine-tuning jobs

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/fine_tuning/jobs' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key'
```

## Get fine-tuning job details

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/fine_tuning/jobs/{job_id}' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key'
```

## Cancel fine-tuning job

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/fine_tuning/jobs/{job_id}/cancel' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key'
```

## List fine-tuning job checkpoints

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/fine_tuning/jobs/{job_id}/checkpoints' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-key'
```

**Note:** Replace `your-secret-key` with the value of your `API_KEY` environment variable.

## Unit Testing Setup in VS Code

To set up and run unit tests in VS Code with pytest support, follow these steps:

### 1. Install Required Extension

Install the **Python Extension Pack** from VS Code extensions marketplace. This provides complete Python development support including testing capabilities.

### 2. Create VS Code Settings File

Create a `.vscode/settings.json` file in your workspace root with the following configuration:

```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "--rootdir=.",
        "resolver/",
        "tests/",
        "."
    ],
    "python.testing.cwd": "${workspaceFolder}",
    "python.defaultInterpreterPath": "/opt/venv/bin/python",
    "python.testing.autoTestDiscoverOnSaveEnabled": true,
    "python.languageServer": "Pylance",
    "python-envs.pythonProjects": [],
    "python.envFile": "${workspaceFolder}/.env.test"
}
```

**Note:** Update `python.defaultInterpreterPath` to match your tt-metal Python environment location.

### 3. Create Test Environment File

Create a `.env.test` file in the project root with the following configuration:

```bash
PYTHONPATH=[path to tt-metal]:[path to tt-media-server]
TT_METAL_PATH=[path to tt-metal]
```

**Note:** Update the paths to match your local environment setup.

### 4. Configure Python Interpreter

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Search for "Python: Select Interpreter"
3. Choose the Python interpreter from your tt-metal environment

### 5. Running and Debugging Tests

Once configured, you should be able to run and debug (all or some specific) tests directly from VS Code. In order to do that you can open the Testing sidebar or open a test file in the editor.

# Configuration

The TT Inference Server can be configured using environment variables or by modifying the settings file. All parameter names should be **UPPERCASED** when used as environment variables.

## General Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `LOG_LEVEL` | `"INFO"` | Sets the logging level for the application. Valid values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `ENVIRONMENT` | `"development"` | Specifies the runtime environment. Used for environment-specific configurations |
| `LOG_FILE` | `None` | Optional path to log file. If not set, logs are output to console only |

## Device Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `DEVICE_IDS` | `"(0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13),(14),(15),(16),(17),(18),(19),(20),(21),(22),(23),(24),(25),(26),(27),(28),(29),(30),(31)"` | Comma-separated list of device IDs available for inference. Defines which TT devices can be used |
| `IS_GALAXY` | `True` | Boolean flag indicating if running on Galaxy hardware. Used for graph device split and class initialization |
| `DEVICE_MESH_SHAPE` | `(1, 1)` | Tuple defining the device mesh topology. Format: `(rows, columns)` for multi-device setups |
| `RESET_DEVICE_COMMAND` | `"tt-smi -r"` | Command used to reset TT devices when needed |
| `RESET_DEVICE_SLEEP_TIME` | `5.0` | Time in seconds to wait after device reset before attempting reconnection |
| `ALLOW_DEEP_RESET` | `False` | Boolean flag to enable deep device reset functionality. When enabled, allows more aggressive device reset operations beyond standard reset procedures |
| `USE_GREEDY_BASED_ALLOCATION` | `True` | Boolean flag to enable greedy-based device allocation strategy. When enabled with single device mesh shape (1,1), automatically allocates all available devices from the system |

## Model Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `MODEL_RUNNER` | [`ModelRunners.TT_SDXL_TRACE.value`](config/constants.py ) | Specifies which model runner implementation to use for inference |
| `MODEL_SERVICE` | `None` | Specifies which model service implementation to use for inference. If not set, the default service for the selected model runner will be used |
| `MODEL_WEIGHTS_PATH` | `""` | Path to the main model weights. Used if `HF_HOME` is not set. |
| `PREPROCESSING_MODEL_WEIGHTS_PATH` | `""` | Path to preprocessing model weights (e.g., for audio preprocessing). Used if `HF_HOME` is not set. |
| `TRACE_REGION_SIZE` | `34541598` | Memory size allocated for model tracing operations (in bytes) |
| `DOWNLOAD_WEIGHTS_FROM_SERVICE` | `True` | Boolean flag to enable downloading weights when initializing service. When enabled, ensures that weights are downloaded once per instance of the server |


## Queue and Batch Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `MAX_QUEUE_SIZE` | `5000` | Maximum number of requests that can be queued for processing |
| `MAX_BATCH_SIZE` | `1` | Maximum batch size for inference requests. Currently limited to 1 for stability |
| `MAX_BATCH_DELAY_TIME_MS` | `None` | Maximum wait time in milliseconds after the first request before a batch is executed, allowing more requests to accumulate without adding significant latency |
| `USE_DYNAMIC_BATCHER` | `False` | Boolean flag to enable dynamic batching for improved throughput. When enabled, the server attempts to batch multiple requests together for more efficient processing |
| `USE_QUEUE_PER_WORKER` | `False` | Boolean flag to enable per-worker result queues. When enabled, each worker has its own dedicated result queue instead of a shared queue, which can improve performance in high-concurrency scenarios by reducing queue contention |
| `QUEUE_FOR_MULTIPROCESSING` | `TTQueue` | Selects the queue implementation for inter-process communication. Options: `TTQueue` (default, Python's multiprocessing.Queue), `FasterFifo` (high-performance, uses faster-fifo library). |

### Dynamic Batching

The `USE_DYNAMIC_BATCHER` setting controls whether the server uses dynamic batching to improve throughput:

- **When `False` (default)**: While one request is in process, new requests are not added
- **When `True`**: The server attempts to add multiple requests during the inference

**Usage:**
```bash
# Enable dynamic batching for higher throughput scenarios
export USE_DYNAMIC_BATCHER=true
export MAX_BATCH_SIZE=4
export MAX_BATCH_DELAY_TIME_MS=50
```

**Note:** Dynamic batching is currently experimental and may not be supported by all model runners. Check your specific model runner documentation for batching support.

## Worker Management

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `NEW_DEVICE_DELAY_SECONDS` | `0` | Delay in seconds before initializing a new device worker, 0 by default |
| `NEW_RUNNER_DELAY_SECONDS` | `2` | Delay in seconds before initializing a new CPU worker |
| `MOCK_DEVICES_COUNT` | `5` | Number of mock devices to create when running in mock/test mode |
| `MAX_WORKER_RESTART_COUNT` | `5` | Maximum number of times a worker can be restarted before being marked as failed |
| `WORKER_CHECK_SLEEP_TIMEOUT` | `30.0` | Time in seconds between worker health checks |
| `DEFAULT_THROTTLE_LEVEL` | `"5"` | Controls the maximum number of concurrent tasks or requests a worker can handle before throttling is applied |

## Timeout Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `REQUEST_PROCESSING_TIMEOUT_SECONDS` | `1000` | Default timeout for processing requests in seconds |

## Job Management Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `MAX_JOBS` | `10000` | Maximum number of jobs allowed in the job manager. |
| `JOB_CLEANUP_INTERVAL_SECONDS` | `300` | Interval in seconds between automatic job cleanup checks. The background cleanup task runs at this frequency to remove old jobs and cancel stuck jobs |
| `JOB_RETENTION_SECONDS` | `86400` | Duration in seconds to keep completed or failed jobs before automatic removal. Jobs older than this threshold are cleaned up to free memory. Default is 1 day |
| `JOB_MAX_STUCK_TIME_SECONDS` | `10800` | Maximum time in seconds a job can remain in "in_progress" status before being automatically cancelled as stuck. Helps prevent zombie jobs from consuming resources. Default is 3 hours |
| `ENABLE_JOB_PERSISTENCE` | `False` | Boolean flag to enable persistent job storage to database. When enabled, jobs are saved to disk and can survive server restarts |
| `JOB_DATABASE_PATH` | `./jobs.db` | The file system path where the job database is stored. This setting is only applicable when job persistence is enabled |

## VLLM Settings

These settings configure VLLM-based model runners and are grouped under `settings.vllm` in the configuration.

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `VLLM__MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | Hugging Face model identifier for VLLM inference. |
| `VLLM__MIN_CONTEXT_LENGTH` | `32` | Sets the minimum number of tokens that can be processed per sequence. Must be a power of two. Must be less than max_model_length. Min value is 32. |
| `VLLM__MAX_MODEL_LENGTH` | `2048` | Sets the maximum number of tokens that can be processed per sequence, including both input and output tokens. Determines the model's context window size. |
| `VLLM__MAX_NUM_BATCHED_TOKENS` | `2048` | Sets the maximum total number of tokens processed in a single iteration across all active sequences. Higher values improve throughput but increase memory usage and latency. |
| `VLLM__MAX_NUM_SEQS` | `1` | Defines the maximum number of sequences that can be batched and processed simultaneously in one iteration. Note: tt-xla currently only supports max_num_seqs=1. |
| `VLLM__GPU_MEMORY_UTILIZATION` | `0.1` | Fraction of GPU memory to use for model weights and KV cache. |

## Audio Processing Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `ALLOW_AUDIO_PREPROCESSING` | `True` | Boolean flag to allow audio preprocessing capabilities |
| `AUDIO_CHUNK_DURATION_SECONDS` | Auto-calculated | Duration in seconds for audio chunks during processing. If not set, automatically calculated based on worker count: 3s for 8+ workers, 15s for 4-7 workers, 30s for 1-3 workers. Can be overridden by setting this environment variable |
| `MAX_AUDIO_DURATION_SECONDS` | `60.0` | Maximum allowed audio duration (in seconds) |
| `MAX_AUDIO_DURATION_WITH_PREPROCESSING_SECONDS` | `300.0` | Maximum allowed audio duration (in seconds) when audio preprocessing (e.g., speaker diarization) is enabled |
| `MAX_AUDIO_SIZE_BYTES` | `52428800` | Maximum allowed audio file size (50 MB in bytes) |
| `DEFAULT_SAMPLE_RATE` | `16000` | Default audio sample rate for processing (16 kHz) |
| `AUDIO_TASK` | `"transcribe"` | Specifies the audio processing task: transcription (speech-to-text in original language) or translation (speech-to-English or other supported language) |
| `AUDIO_LANGUAGE` | `"English"` | Specifies the language for audio processing (transcription or translation). Supported languages depend on the selected Whisper model. |

### Telemetry Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `ENABLE_TELEMETRY` | `True` | Boolean flag to enable or disable telemetry collection. When disabled, no metrics are recorded and background telemetry processes are not started |
| `PROMETHEUS_ENDPOINT` | `"/metrics"` | HTTP endpoint path where Prometheus metrics are exposed for scraping by monitoring systems |

## Authentication Settings

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `API_KEY` | `"your-secret-key"` | Secret key used for API authentication. All requests must include `Authorization: Bearer <API_KEY>` header |

## Hugging Face Configuration

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `HF_TOKEN` | `None` | Hugging Face token with read permission for accessing private models and datasets |
| `HF_HOME` | `None` | Directory path for Hugging Face cache and model storage |

## Special Environment Variable Overrides

The server supports special environment variable combinations that can override multiple settings at once:

| Environment Variable | Description |
|---------------------|-------------|
| `MODEL` | Specifies the model to run. Combined with `DEVICE`, overrides configuration based on predefined ModelConfigs |
| `DEVICE` | Specifies the target device type for model execution. Combined with `MODEL`, overrides configuration based on predefined ModelConfigs |

When both `MODEL` and `DEVICE` are set, the server will look up the corresponding configuration in [`ModelConfigs`](config/constants.py ) and apply all associated settings automatically.

## Telemetry

The TT Media Server provides comprehensive Prometheus metrics for monitoring performance and operational health. Telemetry can be enabled/disabled via the `ENABLE_TELEMETRY` environment variable.

### Available Metrics

#### Request Processing Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `tt_media_server_requests_total` | Counter | Total number of top-level requests | `model_type` |
| `tt_media_server_request_duration_seconds` | Histogram | End-to-end request duration | `model_type` |
| `tt_media_server_requests_base_counter` | Counter | Total base service requests | `model_type` |
| `tt_media_server_requests_base_duration_seconds` | Histogram | Base service request duration | `model_type` |
| `tt_media_server_requests_base_total` | Counter | Total base service method calls | `model_type` |
| `tt_media_server_requests_base_duration_seconds_total` | Histogram | Total base service method duration | `model_type` |

#### Processing Pipeline Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `tt_media_server_pre_processing_duration_seconds` | Histogram | Pre-processing stage duration | `model_type`, `preprocessing_enabled` |
| `tt_media_server_post_processing_duration_seconds` | Histogram | Post-processing stage duration | `model_type`, `post_processing_enabled` |

#### Model & Device Metrics

| Metric Name | Type | Description | Labels |
|-------------|------|-------------|---------|
| `tt_media_server_model_inference_duration_seconds` | Histogram | Model inference execution time | `model_type`, `device_id` |
| `tt_media_server_model_inference_total` | Counter | Total model inference operations | `model_type`, `device_id`, `status` |
| `tt_media_server_device_warmup_duration_seconds` | Histogram | Device warmup time | `model_type`, `device_id` |
| `tt_media_server_device_warmup_total` | Counter | Total device warmup operations | `model_type`, `device_id`, `status` |
| `tt_media_server_model_load_total` | Counter | Total model load operations | `model_type`, `device_id`, `status` |

### Labels Description

Labels are part of the metrics. Example:
tt_media_server_device_warmup_duration_seconds_sum{device_id="2",model_type="tt-sdxl-trace"} 505.4703781604767

- **`model_type`**: The type of model being used (e.g., `SDXL`, `TT_SDXL_IMAGE_TO_IMAGE`)
- **`device_id`**: Identifier for the Tenstorrent device being used
- **`status`**: Operation status (`success` or `failure`)
- **`preprocessing_enabled`**: Whether preprocessing is enabled (`true` or `false`)
- **`post_processing_enabled`**: Whether post-processing is enabled (`true` or `false`)

### Accessing Metrics

Metrics are available at the configured endpoint (default: `http://localhost:8000/metrics`) in Prometheus format.

## Device Mesh Configuration

The server supports special environment variables for configuring device mesh shapes for specific model configurations:

| Environment Variable | Device Mesh Shape | Description |
|---------------------|-------------------|-------------|
| `SD_3_5_FAST` | `None` | Configures device mesh for SD-3.5 in fast configuration (4x8 mesh = 32 devices total) when set to `"true"` (case-insensitive) |
| `SD_3_5_BASE` | `None` | Configures device mesh for SD-3.5 in base configuration (2x4 mesh = 8 devices total) when set to `"true"` (case-insensitive) |
| `TP2` | `None` | Enables tensor parallelism across 2 devices (2x1 mesh) when set to `"true"` (case-insensitive). **Compatible with SDXL models only** |

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
export REQUEST_PROCESSING_TIMEOUT_SECONDS=300
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

# Profiling

We use [py-spy](https://github.com/benfred/py-spy) to profile the server.
To profile the server, first run the media server:

```bash
uvicorn main:app --lifespan on --port 8000
```

The console will print the PID of the server and the worker process PID:
```
INFO:     Started server process [1388662]
2025-12-11 11:58:49,925 - INFO - Started worker 0 with PID 1388679
```

Then run the profiler in two separate terminals, once for the server and once for the worker:
```bash
py-spy record -o profile_server.svg --pid <PID>
py-spy record -o profile_worker.svg --pid <PID>
```

Output is a flame chart [see interactive example](./docs/profiling-example.svg).

How to read the flame chart:

| Color | Width | Meaning | Interpretation | Action Needed |
|-------|-------|---------|----------------|---------------|
| **Light/Green** | **Narrow** | Fast function, quick execution | Efficient code, no issues | Perfect! Ignore it |
| **Light/Green** | **Wide** | I/O bound or coordinator function | Lots of waiting (network, disk, async) or delegates work to many children | Check if waiting is necessary. Optimize I/O if possible |
| **Yellow/Orange** | **Narrow** | Moderate CPU work, short duration | Some computation, but not critical | Monitor, usually okay |
| **Yellow/Orange** | **Wide** | Moderate CPU work, long duration | Doing noticeable work across time | Investigate if it can be optimized |
| **Red/Dark** | **Narrow** | CPU-intensive but quick | Hot code, but doesn't run long | Low priority - fast enough despite intensity |
| **Red/Dark** | **Wide** | CPU-intensive AND long-running | BOTTLENECK! | TOP PRIORITY - Optimize this first! |

# Remaining work:

1. Add unit tests
2. Add API tests
3. Cleanup unused things in runners
