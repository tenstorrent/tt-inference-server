# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TT non-LLM inference server: a FastAPI-based inference server for running non-LLM models (image gen, audio, video, embeddings, CNN) on Tenstorrent hardware. It exposes an OpenAI-compatible API. A separate high-performance C++ implementation lives in `cpp_server/` and has its own `cpp_server/CLAUDE.md`.

## Development Setup

```bash
# Prerequisites
sudo apt update && sudo apt install -y ffmpeg
pip install -r requirements.txt

# Run the server (lifespan required to init/close devices)
uvicorn main:app --lifespan on --port 8000
```

For TP2 (tensor parallelism with 2 devices) or SD-3.5 multi-device configs:
```bash
source run_uvicorn.sh
```

## Running Tests

```bash
# Run all unit tests (mocks hardware deps like ttnn/torch)
pytest tests/

# Run a single test file
pytest tests/test_scheduler.py

# Run a single test
pytest tests/test_scheduler.py::TestScheduler::test_process_request
```

Tests use `tests/conftest.py` which mocks all hardware-dependent modules (`ttnn`, `torch`, `transformers`, `diffusers`, `vllm`, etc.) so they can run without Tenstorrent hardware.

## Configuration

Settings are in `config/settings.py` (pydantic-settings) with defaults in `config/constants.py`. All fields can be overridden via environment variables.

**Preferred setup**: set `MODEL` + `DEVICE` env vars; settings will auto-configure `model_runner`, `device_ids`, `device_mesh_shape`, etc. from `ModelConfigs` in `config/constants.py`:

```bash
export MODEL=flux.1-dev
export DEVICE=t3k
uvicorn main:app --lifespan on --port 8000
```

**Direct runner control**: set `MODEL_RUNNER` directly (skips MODEL/DEVICE lookup):
```bash
export MODEL_RUNNER=tt-sdxl-trace
uvicorn main:app --lifespan on --port 8000
```

`MODEL_SERVICE` is automatically derived from `MODEL_RUNNER` using `MODEL_SERVICE_RUNNER_MAP`. Only the endpoints for the active service are registered.

Key env vars:
- `MODEL` – model name (e.g. `flux.1-dev`, `whisper-large-v3`, `bge-large-en-v1.5`)
- `DEVICE` – device type (`n150`, `n300`, `t3k`, `galaxy`, `p150`, `p300`, `p300x2`, `p150x4`, `p150x8`)
- `MODEL_RUNNER` – runner name (bypasses MODEL+DEVICE lookup)
- `DEVICE_IDS` – explicit device IDs like `(0),(1),(2),(3)`
- `API_KEY` / `OPENAI_API_KEY` – Bearer token for auth (default: `your-secret-key`)
- `HF_TOKEN` – Hugging Face token for model downloads
- `SD_3_5_FAST=true`, `SD_3_5_BASE=true`, `TP2=true` – special mesh configs

## Architecture

### Request Flow

```
HTTP Request
  → FastAPI (main.py)
  → open_ai_api/<service>.py (controller/router)
  → model_services/<service>_service.py (business logic)
  → Scheduler (model_services/scheduler.py) via asyncio queue
  → device_workers/device_worker.py (worker process per device)
  → tt_model_runners/<runner>.py (hardware runner)
```

### Key Layers

- **`open_ai_api/`** – FastAPI routers, one file per service type. Routes registered dynamically in `__init__.py` based on active `MODEL_SERVICE`. Both `/v1/...` (primary) and legacy paths (deprecated, sunset 2026-06-30) are registered.
- **`model_services/`** – Service classes (`ImageService`, `AudioService`, etc.) extend `BaseService`. Handle pre/post processing and segmentation for chunked requests.
- **`resolver/`** – Singleton factories: `service_resolver()` returns the active service; `get_scheduler()` returns the scheduler.
- **`device_workers/`** – Worker processes (`device_worker.py`, `device_worker_dynamic_batch.py`) run in separate processes per device, communicate via queues.
- **`tt_model_runners/`** – Model runner implementations. `runner_fabric.py::get_device_runner()` maps `ModelRunners` enum → runner class. Base class: `BaseDeviceRunner`.
- **`domain/`** – Pydantic request/response models, one file per operation type.
- **`config/constants.py`** – All enums (`ModelRunners`, `ModelServices`, `ModelNames`, `DeviceTypes`) and the `ModelConfigs` dict mapping `(runner, device) → settings`.
- **`config/settings.py`** – Pydantic-settings `Settings` class; singleton `settings` object used throughout.

### Adding a New Model/Runner

1. Add entries to enums in `config/constants.py`: `SupportedModels`, `ModelNames`, `ModelRunners`
2. Map it to a service in `MODEL_SERVICE_RUNNER_MAP` and to model names in `MODEL_RUNNER_TO_MODEL_NAMES_MAP`
3. Add device configs to `ModelConfigs`
4. Implement runner class extending `BaseDeviceRunner` in `tt_model_runners/`
5. Register the runner in `tt_model_runners/runner_fabric.py::AVAILABLE_RUNNERS`
6. Mock the runner in `tests/conftest.py` for unit tests

### API Versioning

All endpoints use `/v1` prefix. Legacy paths (without `/v1`) are deprecated and return `Deprecation: true`, `Sunset: 2026-06-30`, and `Link` headers per RFC 8594/8288. Remove legacy paths after 2026-06-30. Maintenance endpoints (`/tt-liveness`, `/tt-deep-reset`, `/tt-reset-device`) have no version prefix.
