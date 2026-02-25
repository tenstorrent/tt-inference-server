# TT Media Server - C++ Drogon Implementation

A high-performance C++ implementation of the TT Media Server using the Drogon web framework. This implementation is designed to benchmark the overhead of the Python FastAPI server by providing an identical API with minimal overhead.

## LLM engine

The LLM engine lives under `include/runners/llm_engine/` (headers) and `src/runners/llm_engine/` (sources). The engine uses the server’s logging (`[DEBUG] [llm_engine:...]`) instead of its own.

### Main features

- **Paged attention** — KV cache is managed in fixed-size blocks; sequences hold a block table and blocks are allocated or freed as needed. Enables non-contiguous cache and reuse across sequences.
- **Prefix caching** — Content-addressable blocks (hash over token content with optional prefix): when a new sequence shares a prefix with an existing block, the block is reused (reference-counted) so shared prefixes are not recomputed.
- **Prefill-only or decode-only batches** — Each scheduling step returns either a **prefill-only** batch or a **decode-only** batch; there are no mixed prefill+decode batches. Prefill is prioritized over decode when both are possible.
- **Preemption** — When decode cannot proceed (e.g. no free block to extend a sequence), a running sequence can be preempted: its KV cache is freed and it re-enters the waiting queue for later prefill.

The engine does **not** support chunked prefill: each request is prefilled in full when it is scheduled (subject to batch token limits).

**Device backend** — Host–device communication is behind an `IDeviceBackend` abstraction (`init`, `write`, `read`, `terminate`). Two implementations: **mock** (no hardware; echoes written pages back as read data) and **sockets** (TT device, H2D/D2H sockets, loopback kernels). The backend is chosen from `llm_engine::Config::device`, set via `LLM_DEVICE_BACKEND` (see Environment Variables). Default is mock.

### Run unit tests

Run scheduler unit tests (Google Test):

```bash
cd build && ctest --output-on-failure
# or run the test binary directly:
./build/scheduler_test
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
  deepseek-ai/DeepSeek-V3/tokenizer.json
  deepseek-ai/DeepSeek-V3/tokenizer_config.json
  meta-llama/Llama-3.1-8B-Instruct/tokenizer.json
  meta-llama/Llama-3.1-8B-Instruct/tokenizer_config.json
```

Llama models are gated on HuggingFace — set `HF_TOKEN` (or
`HUGGING_FACE_HUB_TOKEN`, or run `huggingface-cli login`) before building to
download them. If the Llama download fails, the build continues (DeepSeek is
required; Llama is optional unless `MODEL_RUNNER=llama_runner`). If both
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
selected at **runtime** via the `MODEL_RUNNER` environment variable — no
recompilation needed:

| `MODEL_RUNNER` | Model | Tokenizer |
|----------------|-------|-----------|
| `llm_test` (default) | DeepSeek V3 | `tokenizers/deepseek-ai/DeepSeek-V3/` |
| `llama_runner` | Llama 3.1 8B Instruct | `tokenizers/meta-llama/Llama-3.1-8B-Instruct/` |

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
| `MODEL_RUNNER` | Runner type: `llm_test` (LLM_TEST, in-process stub) or `llama_runner` (spawns `python -m tt_model_runners.llama_runner`). | `llm_test` |
| `TT_PYTHON_PATH` | Path added to Python `sys.path` for embedding runner (C++ only). | `..` |
| `LLM_DEVICE_BACKEND` | LLM device backend: `sockets` (TT device H2D/D2H) or `mock` (no hardware). | `mock` |
| `OPENAI_API_KEY` | Bearer token for API authentication. | `your-secret-key` |

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
│   │   └── llm_controller.hpp      # OpenAI-compatible API controller
│   ├── domain/
│   │   ├── completion_request.hpp  # Request domain object
│   │   └── completion_response.hpp # Response domain objects
│   ├── runners/
│   │   └── runner_factory.hpp      # Runner factory (env-based selection)
│   ├── scheduler/
│   │   └── multiprocess_scheduler.hpp  # Multiprocess scheduler
│   └── services/
│       ├── base_service.hpp        # Base service class
│       └── llm_service.hpp         # LLM service
├── src/
│   ├── api/
│   │   └── llm_controller.cpp
│   ├── scheduler/
│   │   └── scheduler.cpp
│   ├── services/
│   │   └── base_service.cpp
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
- `RunnerFactory`: Creates appropriate runner based on `TT_RUNNER_TYPE` environment variable

### API
- `LLMController`: Drogon HTTP controller with OpenAI-compatible endpoints

## Runner Types

The server supports the following runner type, selected via the `TT_RUNNER_TYPE` environment variable:

| Runner | Value | Description |
|--------|-------|-------------|
| LLMTestRunner | `llm_test` (default) | Pure CPU benchmark, generates 120k tokens/sec |
| Llama runner | `llama_runner` | Spawns `python -m tt_model_runners.llama_runner` (TTNN device) |

### LLMTestRunner (Default)

Generates tokens at 120,000 tokens/second using busy-wait loops for microsecond precision timing. This allows benchmarking the server infrastructure overhead independent of any device I/O.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | OpenAI-compatible text completion |
| `/health` | GET | Health check |
| `/tt-liveness` | GET | Liveness check with system status |

## Performance

The `LLMTestRunner` is designed to generate tokens at **120,000 tokens/second** using busy-wait loops for microsecond precision timing. This allows benchmarking the server infrastructure overhead independent of actual model inference.

Token generation timing:
- Target: 120,000 tokens/second
- Token interval: ~8.33 microseconds
- Uses `std::chrono::high_resolution_clock` for precise timing

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
4. Tokenizer files are stored per-model under `tokenizers/<model-name>/`. The
   active tokenizer is selected at runtime based on `MODEL_RUNNER` (see
   [Runtime model selection](#runtime-model-selection) above).

## Performance

The `LLMTestRunner` is designed to generate tokens at **120,000 tokens/second** using busy-wait loops for microsecond precision timing. This allows benchmarking the server infrastructure overhead independent of actual model inference.

Token generation timing:
- Target: 120,000 tokens/second
- Token interval: ~8.33 microseconds
- Uses `std::chrono::high_resolution_clock` for precise timing

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
