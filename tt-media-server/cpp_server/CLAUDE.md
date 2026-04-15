# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance C++ implementation of the TT Media Server using the Drogon web framework. It's designed as a benchmark comparison to the Python FastAPI server by providing an identical OpenAI-compatible API with minimal overhead. The server supports LLM chat completions and embeddings with a sophisticated LLM engine featuring paged attention, prefix caching, and preemption.

## Build and Development Commands

### Building the Project

```bash
# Standard release build (tokenizers for all models are pre-fetched automatically)
./build.sh

# Development builds
./build.sh --debug          # Debug build with symbols
./build.sh --asan           # AddressSanitizer + LeakSanitizer for memory debugging
./build.sh --tsan           # ThreadSanitizer for race condition detection
```

Model selection is at runtime via `LLM_DEVICE_BACKEND` env var (default: `mock` = DeepSeek V3; `llama` = Llama 3.1 8B Instruct).

### Running the Server

```bash
# Default (0.0.0.0:8000)
./build/tt_media_server_cpp

# Custom host/port
./build/tt_media_server_cpp -h 127.0.0.1 -p 8001

# Background with logging
nohup ./build/tt_media_server_cpp -p 8001 > server.log 2>&1 &

# Stop background server
pkill -f tt_media_server_cpp
```

### Testing

```bash
# Run all unit tests
cd build && ctest --output-on-failure

# Individual test binaries
./build/scheduler_test        # LLM engine scheduler tests
./build/llm_runner_test       # LLM runner integration tests
./build/sequence_test        # Sequence management tests
./build/test_tokenizer       # Tokenizer functionality tests

# IPC smoke test
./build/ipc_scheduler_smoke_test

# Python integration test (embedding)
python tests/test_embedding.py
```

### Benchmarking

```bash
# Tokenizer performance benchmark
./build/tokenizer_benchmark
```

## Architecture Overview

The server follows a layered architecture mirroring the Python implementation:

### Core Components

- **API Layer**: Drogon HTTP controllers (`api/`) providing OpenAI-compatible endpoints
- **Services Layer**: Business logic (`services/`) handling request processing and validation
- **Workers**: Multiprocess worker architecture (`worker/`) with IPC communication
- **LLM engine**: Inference engine (`runners/llm_runner/`; namespace `tt::runners::llm_engine`) with paged attention; CMake target `llm_runner_lib`
- **Runners**: Multiple runner implementations (`runners/`) for different backends
- **Domain Objects**: Request/response models (`domain/`) matching OpenAI API spec

### LLM Engine Features

The core LLM engine (`include/runners/llm_runner/`) provides:
- **Paged Attention**: KV cache managed in fixed-size blocks with block tables
- **Prefix Caching**: Content-addressable blocks for sharing common prefixes
- **Prefill/Decode Separation**: Separate batch types, prefill prioritized over decode
- **Preemption**: Running sequences can be preempted when resources are needed
- **Multiprocess IPC**: Boost.Interprocess queues for scheduler communication

### Service Modes

The server operates in two modes via `MODEL_SERVICE` environment variable:
- **LLM Mode** (`llm`): Provides `/v1/chat/completions`
- **Embedding Mode** (`embedding`): Provides `/v1/embeddings`

### Runner Types and Model Selection

Runner type and tokenizer are selected via `LLM_DEVICE_BACKEND` environment variable. This selects the
tokenizer strategy (chat template, stop tokens, decode filtering) at runtime via
the `Tokenizer` subclass hierarchy (`utils/tokenizers/tokenizer.hpp`):

- **`mock`** or **`pipeline`** (default when unset: `mock`): Mock or TT device runner, DeepSeek V3 tokenizer strategy
- **`llama`**: Python-based runner via pybind11, Llama 3.1 8B Instruct tokenizer strategy

## Configuration System

Configuration follows the same pattern as the Python server - defaults in `config/constants.hpp` with environment variable overrides in `config/settings.hpp`.

### Key Environment Variables

- `MODEL_SERVICE`: `llm` or `embedding` (default: `llm`)
- `LLM_DEVICE_BACKEND`: `mock` or `pipeline` (DeepSeek V3), `llama` (Llama 3.1 8B Instruct) — selects runner + tokenizer strategy (default: `mock`)
- `DEVICE_IDS`: Bracket-pair device list like `(0,1,2,3),(4,5,6,7)` defining workers
- `MAX_BATCH_DELAY_TIME_MS`: Max wait time to fill batches
- `OPENAI_API_KEY`: Bearer token for API authentication (default: `your-secret-key`)

## Development Patterns

### Adding New Endpoints

1. Define request/response objects in `include/domain/`
2. Implement controller in `include/api/` and `src/api/`
3. Add service logic in `include/services/` and `src/services/`
4. Register in service factory (`utils/service_factory.cpp`)

### Testing Strategy

- **Unit Tests**: Google Test for individual components (`tests/*.cpp`)
- **Integration Tests**: Full API testing with Python scripts (`tests/*.py`)
- **IPC Tests**: Multiprocess communication testing
- **Benchmarking**: Performance measurement with precise timing

### Memory Safety

The project uses modern C++20 with strict compiler warnings and sanitizer support:
- Use `./build.sh --asan` for memory leak detection during development
- Use `./build.sh --tsan` for race condition detection
- All tests should pass with sanitizers enabled

### Logging

- Main server uses Drogon logging (in `./logs/` directory)
- LLM engine uses structured logging with a `runners.llm_engine:` tag in `LLM_ENGINE_LOG` output when `LLM_ENGINE_DEBUG_BUILD` is enabled
- Enable LLM engine debug logging with `-DLLM_ENGINE_DEBUG_BUILD=ON`

## Dependencies and Prerequisites

### Required
- **CMake** >= 3.19
- **Drogon Framework** >= 1.8 (automatically installed by build script if missing)
- **C++20 compatible compiler** (GCC 10+, Clang 12+)
- **Boost** (headers for Boost.Interprocess IPC)
- **JsonCpp** (tokenizer config parsing)
- **Rust/Cargo** (required for tokenizers-cpp dependency)

### Optional
- **Python 3** + **ttnn** (for TTNN runner support)
- **AddressSanitizer/ThreadSanitizer** (for debugging builds)

### Tokenizer Support

The build script pre-fetches tokenizer files for all supported models into
per-model subdirectories under `tokenizers/`:
- `tokenizers/deepseek-ai/DeepSeek-R1-0528/tokenizer.json` + `tokenizer_config.json`
- `tokenizers/meta-llama/Llama-3.1-8B-Instruct/tokenizer.json` + `tokenizer_config.json`

The active tokenizer is selected at runtime based on `LLM_DEVICE_BACKEND`. To add a
new model, manually download tokenizer files into `tokenizers/<org>/<model>/`.

## Performance Characteristics

- **Target Throughput**: 120,000 tokens/second (LLMTestRunner)
- **Token Timing**: ~8.33 microseconds per token using high-resolution timing
- **Memory Management**: Block-based KV cache with reference counting
- **Concurrency**: Multiprocess worker architecture with IPC queues
- **Streaming**: Server-Sent Events (SSE) for real-time token streaming

## API Compatibility

The server provides OpenAI-compatible endpoints:
- `POST /v1/chat/completions` - Chat completion with streaming support
- `POST /v1/embeddings` - Text embeddings (embedding mode only)
- `GET /health` - Health check
- `GET /tt-liveness` - Liveness check with detailed system status
- `GET /docs` - Swagger UI documentation
- `GET /openapi.json` - OpenAPI specification

All endpoints except health/tt-liveness/docs require Bearer token authentication.
