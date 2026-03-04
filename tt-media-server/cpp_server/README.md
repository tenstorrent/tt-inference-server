# TT Media Server - C++ Drogon Implementation

A high-performance C++ implementation of the TT Media Server using the Drogon web framework. This implementation is designed to benchmark the overhead of the Python FastAPI server by providing an identical API with minimal overhead.

## LLM engine

The LLM engine lives under `include/runners/llm_runner/` (headers) and `src/runners/llm_runner/` (sources). The engine uses the server’s logging (`[DEBUG] [llm_engine:...]`) instead of its own.

### Main features

- **Paged attention** — KV cache is managed in fixed-size blocks; sequences hold a block table and blocks are allocated or freed as needed. Enables non-contiguous cache and reuse across sequences.
- **Prefix caching** — Content-addressable blocks (hash over token content with optional prefix): when a new sequence shares a prefix with an existing block, the block is reused (reference-counted) so shared prefixes are not recomputed.
- **Prefill-only or decode-only batches** — Each scheduling step returns either a **prefill-only** batch or a **decode-only** batch; there are no mixed prefill+decode batches. Prefill is prioritized over decode when both are possible.
- **Preemption** — When decode cannot proceed (e.g. no free block to extend a sequence), a running sequence can be preempted: its KV cache is freed and it re-enters the waiting queue for later prefill.

The engine does **not** support chunked prefill: each request is prefilled in full when it is scheduled (subject to batch token limits).

**Device backend** — Host–device communication is behind an `IDeviceBackend` abstraction (`init`, `write`, `read`, `terminate`). Two implementations: **mock** (no hardware; echoes written pages back as read data) and **sockets** (TT device, H2D/D2H sockets, loopback kernels). The backend is chosen from `llm_engine::Config::device`, set via `LLM_DEVICE_BACKEND` (see Environment Variables). Default is mock.

### Run unit tests

Run LLM engine unit tests (Google Test) from the `cpp_server` directory:

```bash
cd build && ctest --output-on-failure
# Or run test binaries directly:
./build/scheduler_test
./build/llm_runner_test
./build/sequence_test
./build/ipc_scheduler_smoke_test
```

## Quick Start

```bash
# Build (defaults to DeepSeek V3)
cd cpp_server
./build.sh

# Start the server (foreground)
./build/tt_media_server_cpp -p 8001

# Or start in background
./build/tt_media_server_cpp -p 8001 &

# Test it
curl http://localhost:8001/health

# Stop the server (if running in background)
pkill -f tt_media_server_cpp
# Or use Ctrl+C if running in foreground
```

## Build Options

```bash
# Default build
./build.sh

# Debug build
./build.sh --debug

# AddressSanitizer
./build.sh --asan
```

### Tokenizer files

The build script automatically pre-fetches tokenizer files for all supported
models from HuggingFace into `tokenizers/<model-name>/`:

```
tokenizers/
  deepseek-ai/DeepSeek-R1-0528/tokenizer.json
  deepseek-ai/DeepSeek-R1-0528/tokenizer_config.json
  meta-llama/Llama-3.1-8B-Instruct/tokenizer.json
  meta-llama/Llama-3.1-8B-Instruct/tokenizer_config.json
```

Llama models are gated on HuggingFace — set `HF_TOKEN` (or
`HUGGING_FACE_HUB_TOKEN`, or run `huggingface-cli login`) before building to
download them. If the Llama download fails, the build continues (DeepSeek is
required; Llama is optional unless `LLM_DEVICE_BACKEND=llama`). If both
`tokenizer.json` and `tokenizer_config.json` already exist for a model, the
build skips the download (no `HF_TOKEN` needed for subsequent builds). To force
re-download, remove the model directory under `tokenizers/<org>/<model>/`.

To add a new model, manually download its tokenizer files into a subdirectory
matching the HuggingFace model name:

```bash
mkdir -p tokenizers/<org>/<model>
wget -O tokenizers/<org>/<model>/tokenizer.json \
  https://huggingface.co/<org>/<model>/raw/main/tokenizer.json
wget -O tokenizers/<org>/<model>/tokenizer_config.json \
  https://huggingface.co/<org>/<model>/raw/main/tokenizer_config.json
```

### Runtime model selection

Model-specific behavior (chat template, stop tokens, decode filtering) is
selected at **runtime** via the `LLM_DEVICE_BACKEND` environment variable — no
recompilation needed:

| `LLM_DEVICE_BACKEND` | Model | Tokenizer |
|----------------------|-------|-----------|
| `mock` or `ttrun` (default when unset: `mock`) | DeepSeek V3 | `tokenizers/deepseek-ai/DeepSeek-R1-0528/` |
| `llama` | Llama 3.1 8B Instruct | `tokenizers/meta-llama/Llama-3.1-8B-Instruct/` |

The runtime selection uses an OOP strategy pattern — see
`include/utils/tokenizer_strategy.hpp` for the `ITokenizerStrategy` interface
and `create_tokenizer_strategy()` factory.

## Starting the Server

### Basic Usage

```bash
# Start with default settings (0.0.0.0:8000, auto-detect threads)
./build/tt_media_server_cpp

# Start on a specific port
./build/tt_media_server_cpp -p 8001

# Start with custom host and port
./build/tt_media_server_cpp -h 127.0.0.1 -p 8080

# Start with specific number of IO threads
./build/tt_media_server_cpp -t 16

# Show help
./build/tt_media_server_cpp --help
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-h, --host HOST` | Listen address | `0.0.0.0` |
| `-p, --port PORT` | Listen port | `8000` |
| `-t, --threads N` | Number of IO threads | CPU cores |
| `--help` | Show help message | - |

### Environment Variables

Configuration is read via `config/settings.hpp` (defaults with env overrides, similar to `tt-media-server/config/settings.py`). No direct `getenv` elsewhere.

| Variable | Description | Default |
|----------|-------------|---------|
| `DEVICE_IDS` | Bracket-pair device list, one worker per pair (e.g. `(0,1,2,3),(4,5,6,7)`). num_workers = number of pairs; each worker's `TT_VISIBLE_DEVICES` = that pair's contents. | `(0),(1),(2),(3)` |
| `MODEL_SERVICE` | Service mode: `embedding` or `llm`. Same as tt-media-server. | `llm` |
| `MAX_BATCH_SIZE` | Max requests per batch (embedding). Same as tt-media-server. | `1` |
| `MAX_BATCH_DELAY_TIME_MS` | Max wait (ms) to fill batch (embedding). Same as tt-media-server. | `5` |
| `TT_PYTHON_PATH` | Path added to Python `sys.path` for embedding runner (C++ only). | `..` |
| `LLM_DEVICE_BACKEND` | LLM device backend and model: `mock` or `ttrun` (DeepSeek V3 tokenizer), `llama` (Llama 3.1 8B Instruct). | `mock` |
| `OPENAI_API_KEY` | Bearer token for API authentication. | `your-secret-key` |
| `LLM_MODE` | LLM operating mode: `regular`, `prefill`, or `decode`. See Prefill/Decode Split Mode. | `regular` |
| `SOCKET_HOST` | Socket host for prefill/decode communication. Decode server: bind address. Prefill server: decode server address. | `localhost` |
| `SOCKET_PORT` | Socket port for prefill/decode communication. | `9000` |

### Prefill/Decode Split Mode

The server supports running in a split architecture where prefill and decode operations are handled by separate server instances on different machines. This enables distributing the workload across multiple nodes.

**Modes:**
- `regular` (default): Single server handles both prefill and decode
- `prefill`: Server only performs prefill (processes prompt, generates first token)
- `decode`: Server receives HTTP requests, forwards to prefill server, then generates remaining tokens locally

**Architecture:**
```
Client HTTP Request
        │
        ▼
┌───────────────────┐
│   Decode Server   │ (LLM_MODE=decode, port 8001)
│   Socket Server   │ (SOCKET_PORT=9000)
└─────────┬─────────┘
          │ TCP Socket
          ▼
┌───────────────────┐
│  Prefill Server   │ (LLM_MODE=prefill, port 8002)
│   Socket Client   │ (connects to decode server)
└───────────────────┘
```

**Running the split architecture:**

1. **Start Decode Server** (receives HTTP requests, listens for prefill connections):
   ```bash
   LLM_MODE=decode SOCKET_HOST=0.0.0.0 SOCKET_PORT=9000 \
     ./build/tt_media_server_cpp -p 8001
   ```

2. **Start Prefill Server** (connects to decode server):
   ```bash
   LLM_MODE=prefill SOCKET_HOST=<decode-server-ip> SOCKET_PORT=9000 \
     ./build/tt_media_server_cpp -p 8002
   ```

3. **Send requests to the Decode Server**:
   ```bash
   curl -X POST http://<decode-server>:8001/v1/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-secret-key" \
     -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'
   ```

**Flow:**
1. Client sends request to decode server (HTTP)
2. Decode server sends prefill request to prefill server (socket)
3. Prefill server processes prompt, generates first token, sends prefill result with token IDs
4. Decode server continues generating remaining tokens locally
5. Decode server streams response to client

### Logging Configuration

The C++ server uses spdlog for structured, high-performance logging. Logging is configured through environment variables:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SPDLOG_LEVEL` | Log level: `trace`, `debug`, `info`, `warn`, `error`, `critical`, `off` | `info` | `SPDLOG_LEVEL=debug` |
| `TT_LOG_FILE` | Log file path (optional, defaults to console only) | - | `TT_LOG_FILE=./logs/server.log` |
| `TT_LOG_MAX_SIZE` | Maximum log file size in MB before rotation | `10` | `TT_LOG_MAX_SIZE=50` |
| `TT_LOG_MAX_FILES` | Number of rotated log files to keep | `3` | `TT_LOG_MAX_FILES=5` |

#### Usage in Code

```cpp
#include "utils/logger.hpp"

// Using the singleton logger
auto logger = tt::utils::Logger::get_logger();
logger->log_info("Server starting on port {}", 8001);
logger->log_error("Failed to connect: {}", error_message);

// Using convenience macros (recommended)
TT_LOG_INFO("Request processed in {} ms", duration);
TT_LOG_DEBUG("Debug info: user={}, request_id={}", user, req_id);
TT_LOG_WARN("High memory usage: {} MB", memory_usage);
TT_LOG_ERROR("Database connection failed: {}", db_error);
```

#### Examples

```bash
# Console logging only (default)
./build/tt_media_server_cpp

# Console + file logging with debug level
SPDLOG_LEVEL=debug TT_LOG_FILE=./server.log ./build/tt_media_server_cpp

# File logging with rotation (50MB files, keep 10)
TT_LOG_FILE=./logs/server.log TT_LOG_MAX_SIZE=50 TT_LOG_MAX_FILES=10 ./build/tt_media_server_cpp

# Silent mode (errors only)
SPDLOG_LEVEL=error ./build/tt_media_server_cpp
```

### Tracy profiling (Tracy build only)

When built with Tracy, use the **C++ Server [CodeLLDB + Tracy]** launch config, then connect the Tracy Profiler UI to **localhost:8086** (main) or **localhost:8087**, **8088**, … (workers). Workers are started via fork+exec so each runs in a fresh process and starts its own Tracy listener.

See [TRACY.md](TRACY.md) for building the GUI, remote port forwarding, and launch configs.

## Authentication

The server uses Bearer token authentication for protected API endpoints. The token is read from the `OPENAI_API_KEY` environment variable at startup. If not set, it defaults to `your-secret-key`.

### Unprotected Endpoints

The following endpoints do not require authentication:
- `GET /health`
- `GET /tt-liveness`
- `GET /docs`
- `GET /swagger`
- `GET /openapi.json`

### Running in Background

```bash
# Start in background with nohup (persists after terminal close)
nohup ./build/tt_media_server_cpp -p 8001 > server.log 2>&1 &

# Start in background (simple)
./build/tt_media_server_cpp -p 8001 &

# Check if running
pgrep -f tt_media_server_cpp

# View logs
tail -f server.log
```

## Stopping the Server

```bash
# If running in foreground: Press Ctrl+C

# If running in background:
pkill -f tt_media_server_cpp

# Or find the PID and kill it
pgrep -f tt_media_server_cpp
kill <PID>

# Force kill if needed
pkill -9 -f tt_media_server_cpp
```

## API Endpoints

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/v1/completions` | POST | ✅ Yes | OpenAI-compatible text completion |
| `/v1/chat/completions` | POST | ✅ Yes | OpenAI-compatible chat completion |
| `/health` | GET | ❌ No | Health check |
| `/tt-liveness` | GET | ❌ No | Liveness check with system status |
| `/docs` | GET | ❌ No | Swagger UI documentation |
| `/openapi.json` | GET | ❌ No | OpenAPI specification |

## Usage Examples

### Non-streaming Completion

```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "max_tokens": 100,
    "stream": false
  }'
```

**Response:**
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "test-model",
  "choices": [
    {
      "text": "token_0 token_1 token_2 ...",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 100,
    "total_tokens": 100
  }
}
```

### Streaming Completion (SSE)

```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, world!",
    "max_tokens": 10,
    "stream": true
  }' --no-buffer
```

**Response (Server-Sent Events):**
```
data: {"id":"cmpl-abc123","choices":[{"text":"token_0","index":0}],...}

data: {"id":"cmpl-abc123","choices":[{"text":"token_1","index":1}],...}

...

data: [DONE]
```

### Health Check

```bash
curl http://localhost:8001/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1234567890
}
```

### Liveness Check

```bash
curl http://localhost:8001/tt-liveness
```

**Response:**
```json
{
  "model_ready": true,
  "queue_size": 0,
  "max_queue_size": 10000,
  "device": "cpu",
  "workers": [
    {
      "worker_id": "worker_0",
      "is_ready": true,
      "processed_requests": 42
    }
  ]
}
```

### Swagger UI

Open in browser: http://localhost:8001/docs

## Architecture

The C++ server mirrors the Python implementation's architecture:

```
cpp_server/
├── include/
│   ├── api/
│   │   ├── llm_controller.hpp       # OpenAI-compatible LLM API
│   │   ├── embedding_controller.hpp
│   │   └── health_controller.hpp
│   ├── config/
│   │   ├── settings.hpp             # Env-based config (defaults + overrides)
│   │   └── constants.hpp            # Default env values, ModelType, etc.
│   ├── domain/
│   │   ├── completion_request.hpp
│   │   ├── completion_response.hpp
│   │   ├── chat_completion_*.hpp
│   │   └── embedding_*.hpp
│   ├── runners/
│   │   ├── llm_runner.hpp           # LLMRunner (scheduler + model runner)
│   │   ├── llm_runner/              # LLM engine (config, scheduler, block manager, model_runner)
│   │   │   ├── config.hpp           # Config, DeviceBackend, ModelRunnerType
│   │   │   ├── model_runner.hpp     # IModelRunner, make_model_runner()
│   │   │   ├── device_backend.hpp  # IDeviceBackend, make_device_backend()
│   │   │   └── ...
│   │   ├── llama_model_runner.hpp   # LlamaModelRunner (pybind11 in-process)
│   │   ├── embedding_runner.hpp
│   │   └── runner_interface.hpp
│   ├── utils/
│   │   ├── runner_factory.hpp       # create_runner() (env-based selection)
│   │   └── tokenizer_strategy.hpp  # LLM_DEVICE_BACKEND → tokenizer
│   ├── services/
│   │   ├── llm_service.hpp
│   │   └── embedding_service.hpp
│   └── worker/
│       └── single_process_worker.hpp
├── src/
│   ├── api/
│   ├── config/
│   ├── runners/
│   │   ├── llm_runner.cpp
│   │   ├── llm_runner/              # model_runner, device_backend, scheduler, ...
│   │   ├── llama_model_runner.cpp
│   │   └── embedding_runner.cpp
│   ├── utils/
│   │   └── runner_factory.cpp       # create_runner() → LLMRunner or EmbeddingRunner
│   ├── services/
│   └── main.cpp
└── CMakeLists.txt
```

## Components

### Domain Objects
- `CompletionRequest` / `CompletionResponse`: OpenAI-compatible completion request and response
- `StreamingChunkResponse`: SSE streaming chunk (completions)
- `ChatCompletionRequest` / `ChatCompletionResponse`: Chat completions request and non-streaming response
- `ChatCompletionStreamChunk`: Chat completions SSE streaming chunk

### Scheduler
- `ThreadSafeQueue<T>`: Lock-free thread-safe queue for task management
- `Scheduler`: Manages worker threads and task distribution

### Services
- `BaseService`: Base class with pre/post processing hooks
- `LLMService`: LLM-specific service implementation

### Runners
- **Runner factory** (`utils/runner_factory.cpp`): Creates the runner based on `MODEL_SERVICE` and `LLM_DEVICE_BACKEND`. For LLM, builds `llm_engine::Config` (including `model_runner` and `device` from config/settings) and passes it to `LLMRunner`; the model runner (stub or Llama pybind11) is created inside the engine via `make_model_runner(config)` (see `include/runners/llm_runner/config.hpp` and `model_runner.cpp`).

### API
- `LLMController`: Drogon HTTP controller with OpenAI-compatible endpoints

## Runner Types

The server supports the following runner types, selected via the `LLM_DEVICE_BACKEND` environment variable:

| Runner | Value | Description |
|--------|-------|-------------|
| Mock / TtRun | `mock` or `ttrun` (default when unset: `mock`) | Mock: no device. TtRun: TT device. Both use DeepSeek V3 tokenizer. |
| Llama runner | `llama` | In-process pybind11: embeds Python and calls `tt_model_runners.llama_runner.Llama31_8BRunner` (TT device). Requires `TT_METAL_HOME`, `HF_MODEL`, tokenizer under `tokenizers/meta-llama/Llama-3.1-8B-Instruct/`. |

### LLM mock runner (default)

When `LLM_DEVICE_BACKEND` is unset or `mock`, the engine uses a stub model runner (no real device). Useful for testing the server and API without hardware.

## Performance

With `LLM_DEVICE_BACKEND=mock`, the stub runner can be used to benchmark server overhead (no device I/O). Real throughput with `llama` depends on the TT device and model.

## Building

### Prerequisites

1. **CMake** >= 3.16
2. **Drogon Framework** >= 1.8
3. **C++20 compatible compiler** (GCC 10+, Clang 12+)
4. **Boost** (headers; used for Boost.Interprocess in the LLM engine IPC queue).
5. **JsonCpp** (used for tokenizer_config parsing).

### Install Drogon (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt install git gcc g++ cmake libjsoncpp-dev uuid-dev \
     openssl libssl-dev zlib1g-dev libbrotli-dev libboost-dev

# Clone and build Drogon
git clone https://github.com/drogonframework/drogon
cd drogon
git submodule update --init
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
```

### Build the Server

```bash
cd cpp_server
chmod +x build.sh
./build.sh           # Release build
./build.sh --debug   # Debug build
./build.sh --asan    # AddressSanitizer + LeakSanitizer (memory/leak detection)
./build.sh --tsan    # ThreadSanitizer (data-race detection; cannot combine with --asan)
```

### Memory leak detection

1. **AddressSanitizer (ASan) + LeakSanitizer (LSan)** — recommended on macOS and Linux:
   ```bash
   ./build.sh --asan
   cd build && ctest --output-on-failure
   # Or run the server and exit; at exit ASan will report leaks and address errors.
   ./tt_media_server_cpp -p 8001
   ```
   Leak reports appear at process exit. Use `LSAN_OPTIONS=verbosity=1` for more detail.

2. **Valgrind** (Linux only; not supported on current macOS):
   ```bash
   valgrind --leak-check=full --show-leak-kinds=all ./build/tt_media_server_cpp -p 8001
   # Or for unit tests:
   valgrind --leak-check=full ./build/scheduler_test
   ```
   Build a normal (non-ASan) binary; Valgrind instruments at runtime.
### Tokenizer (mlc-ai/tokenizers-cpp)

The server includes tokenizer support for encode/decode:

1. Install [Rust](https://rustup.rs) (required by tokenizers-cpp).
2. tokenizers-cpp is **fetched at configure time** via CMake FetchContent. CMake will download it into `build/_deps/`.
3. Build the server — tokenizer files are pre-fetched automatically by `build.sh`:
   ```bash
   ./build.sh
   ```
<<<<<<< HEAD
4. Tokenizer files are stored per-model under `tokenizers/<model-name>/`. The
   active tokenizer is selected at runtime based on `LLM_DEVICE_BACKEND` (see
   [Runtime model selection](#runtime-model-selection) above).
=======
4. Place a HuggingFace `tokenizer.json` (or SentencePiece `tokenizer.model`) at `cpp_server/tokenizers/tokenizer.json`, and `tokenizer_config.json` at `cpp_server/tokenizers/tokenizer_config.json`. The server loads them automatically from those paths relative to the executable.
   To fetch DeepSeek R1 0528 tokenizer and config from Hugging Face into `tokenizers/`:
   ```bash
   mkdir -p cpp_server/tokenizers
   wget -q -O cpp_server/tokenizers/tokenizer.json https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/resolve/main/tokenizer.json
   wget -q -O cpp_server/tokenizers/tokenizer_config.json https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/resolve/main/tokenizer_config.json
   ```

## Performance

The `LLMTestRunner` is designed to generate tokens at **120,000 tokens/second** using busy-wait loops for microsecond precision timing. This allows benchmarking the server infrastructure overhead independent of actual model inference.

Token generation timing:
- Target: 120,000 tokens/second
- Token interval: ~8.33 microseconds
- Uses `std::chrono::high_resolution_clock` for precise timing
>>>>>>> dev

## Comparison with Python FastAPI

| Feature | Python FastAPI | C++ Drogon |
|---------|---------------|------------|
| Framework | FastAPI + Uvicorn | Drogon |
| Async Model | asyncio | epoll/kqueue + threads |
| JSON Library | Pydantic | jsoncpp |
| Queue | multiprocessing.Queue | std::queue + mutex |
| Target Throughput | Variable | 120k tokens/sec |

## Performance Testing

To compare with the Python server:

1. Start the C++ server:
   ```bash
   ./build/tt_media_server_cpp -p 8001
   ```

2. Start the Python server:
   ```bash
   cd .. && python main.py --port 8000
   ```

3. Run load tests against both servers and compare:
   - Request latency
   - Streaming chunk latency (time between tokens)
   - CPU utilization
   - Memory usage

## Troubleshooting

### Server won't start

1. **Port already in use:**
   ```bash
   # Check what's using the port
   lsof -i :8001
   # Kill it or use a different port
   ./build/tt_media_server_cpp -p 8002
   ```

2. **Permission denied:**
   ```bash
   chmod +x ./build/tt_media_server_cpp
   ```

3. **Missing libraries:**
   ```bash
   # Check for missing shared libraries
   ldd ./build/tt_media_server_cpp
   # Install Drogon if missing
   sudo ldconfig
   ```

### Build fails

1. **Drogon not found:**
   ```bash
   # Install Drogon system-wide
   cd /path/to/drogon/build
   sudo make install
   sudo ldconfig
   ```

2. **CMake too old:**
   ```bash
   cmake --version  # Need 3.16+
   ```

### Server crashes

Check the logs in `./logs/` directory or stderr output for error messages.
