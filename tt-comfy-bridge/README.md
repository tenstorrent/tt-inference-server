# TT-Comfy Bridge Server

A Unix domain socket-based bridge server that exposes Tenstorrent tt-metal models to ComfyUI.

## Architecture

The bridge server wraps existing tt-metal model runners from `tt-inference-server/tt-media-server` and provides a low-latency IPC interface for ComfyUI to use Tenstorrent hardware acceleration.

## Features

- Unix socket IPC for <5ms overhead
- Shared memory tensor transfer for zero-copy operations
- Supports SDXL, SD3.5, SD1.4 models
- Image-to-image workflows
- Async operation handling

## Installation

```bash
cd tt-comfy-bridge
pip install -r requirements.txt
```

## Usage

Start the bridge server:

```bash
python -m server.main --device-id 0
```

The server will create a Unix socket at `/tmp/tt-comfy.sock` by default.

## Components

- `server/` - Unix socket server implementation
- `models/` - Model wrappers around tt-metal runners
- `protocol/` - IPC protocol definitions
- `tests/` - Test suite

## Development

Run tests:

```bash
pytest tests/
```

## License

Apache-2.0

