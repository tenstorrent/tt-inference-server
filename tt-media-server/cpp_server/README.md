# TT Media Server - C++ Drogon Implementation

A high-performance C++ implementation of the TT Media Server using the Drogon web framework. This implementation is designed to benchmark the overhead of the Python FastAPI server by providing an identical API with minimal overhead.

## Quick Start

```bash
# Build
cd cpp_server
./build.sh Release

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

| Variable | Description | Default |
|----------|-------------|---------|
| `TT_RUNNER_TYPE` | Runner type: `llm_test` or `ttnn_test` | `llm_test` |

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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | OpenAI-compatible text completion |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check with system status |
| `/docs` | GET | Swagger UI documentation |
| `/openapi.json` | GET | OpenAPI specification |

## Usage Examples

### Non-streaming Completion

```bash
curl -X POST http://localhost:8001/v1/completions \
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

### Readiness Check

```bash
curl http://localhost:8001/ready
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
│   │   ├── base_device_runner.hpp  # Base runner interface
│   │   ├── llm_test_runner.hpp     # Test runner (120k tokens/sec)
│   │   ├── ttnn_test_runner.hpp    # TTNN device I/O runner
│   │   └── runner_factory.hpp      # Runner factory (env-based selection)
│   ├── scheduler/
│   │   ├── scheduler.hpp           # Task scheduler
│   │   └── thread_safe_queue.hpp   # Thread-safe queue
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
- `CompletionRequest`: OpenAI-compatible completion request
- `CompletionResponse`: Full completion response
- `StreamingChunkResponse`: SSE streaming chunk

### Scheduler
- `ThreadSafeQueue<T>`: Lock-free thread-safe queue for task management
- `Scheduler`: Manages worker threads and task distribution

### Services
- `BaseService`: Base class with pre/post processing hooks
- `LLMService`: LLM-specific service implementation

### Runners
- `BaseDeviceRunner`: Abstract base class for model runners
- `LLMTestRunner`: Test runner generating **120,000 tokens/second**
- `TTNNTestRunner`: TTNN device I/O runner (reads tensor from device per token)
- `RunnerFactory`: Creates appropriate runner based on `TT_RUNNER_TYPE` environment variable

### API
- `LLMController`: Drogon HTTP controller with OpenAI-compatible endpoints

## Runner Types

The server supports multiple runner types, selected via the `TT_RUNNER_TYPE` environment variable:

| Runner | Value | Description |
|--------|-------|-------------|
| LLMTestRunner | `llm_test` (default) | Pure CPU benchmark, generates 120k tokens/sec |
| TTNNTestRunner | `ttnn_test` | TTNN device I/O, reads tensor from device per token |

### LLMTestRunner (Default)

Generates tokens at 120,000 tokens/second using busy-wait loops for microsecond precision timing. This allows benchmarking the server infrastructure overhead independent of any device I/O.

### TTNNTestRunner (TTNN Build Required)

Interfaces with the TTNN Python library via embedded Python to measure device I/O overhead:
- Opens a mesh device with shape (1,1)
- Writes a tensor to the device on initialization
- Reads the tensor from the device for each token generated
- Measures real device read latency per token

To use the TTNN runner:

```bash
# Build with TTNN support
./build.sh --ttnn

# Start with TTNN runner
TT_RUNNER_TYPE=ttnn_test ./build/tt_media_server_cpp -p 8001
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | OpenAI-compatible text completion |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check with system status |

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

### Install Drogon (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt install git gcc g++ cmake libjsoncpp-dev uuid-dev \
     openssl libssl-dev zlib1g-dev libbrotli-dev

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
./build.sh --ttnn    # Enable TTNN test runner (requires Python + ttnn)
```

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
