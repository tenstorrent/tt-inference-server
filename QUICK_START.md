# TT-Inference-Server Quick Start Guide

## Overview

TT-Inference-Server provides pre-configured vLLM inference servers for Tenstorrent hardware using Docker containers or local setup.

## Prerequisites

- **Python 3.8+** (Python 3.8.10 on Ubuntu 20.04)
- **python3-venv**: `apt install python3-venv`
- **Docker** (for --docker-server mode)
- **Tenstorrent Hardware**: n150, n300, t3k, galaxy, etc.
- **HuggingFace Token**: Get from https://huggingface.co/settings/tokens

## Simplest Example (Llama-3.2-1B on n150)

```bash
# Run server with Docker (easiest method)
python3 run.py \
  --model Llama-3.2-1B-Instruct \
  --device n150 \
  --workflow server \
  --docker-server
```

On first run, you'll be prompted for:
- **HF_TOKEN**: Your HuggingFace access token
- **JWT_SECRET**: Any secret string for authentication

## Available Models

| Model | Hardware | Min Disk | Status |
|-------|----------|----------|--------|
| Llama-3.2-1B-Instruct | n150, n300, t3k | 4GB | 🟡 Functional |
| Llama-3.2-3B-Instruct | n150, n300, t3k | 12GB | 🟡 Functional |
| Llama-3.1-8B-Instruct | n150, n300, t3k | 32GB | 🟡 Functional |
| Llama-3.1-70B-Instruct | t3k, galaxy | 350GB | 🟡 Functional |
| Mistral-7B-Instruct-v0.3 | n150, n300, t3k | 28GB | 🟡 Functional |
| Qwen2.5-7B-Instruct | n300 | 28GB | 🛠️ Experimental |

See [README.md](README.md) for the complete list.

## Workflows

### 1. Server Only
Start the inference server and leave it running:

```bash
python3 run.py \
  --model Llama-3.2-1B-Instruct \
  --device n150 \
  --workflow server \
  --docker-server
```

The server will run in detached mode. View logs:
```bash
docker ps  # Find container ID
docker logs -f <container-id>
```

### 2. Benchmarks
Run performance benchmarks:

```bash
python3 run.py \
  --model Llama-3.2-1B-Instruct \
  --device n150 \
  --workflow benchmarks \
  --docker-server
```

Results saved to: `workflow_logs/benchmarks_output/`

### 3. Evaluations
Run accuracy evaluations:

```bash
python3 run.py \
  --model Llama-3.2-1B-Instruct \
  --device n150 \
  --workflow evals \
  --docker-server
```

Results saved to: `workflow_logs/evals_output/`

### 4. Release (All Workflows)
Run benchmarks + evals + reports:

```bash
python3 run.py \
  --model Llama-3.2-1B-Instruct \
  --device n150 \
  --workflow release \
  --docker-server
```

## Testing the Server

After the server is running, test it with the prompt client:

```bash
# Activate the workflow venv
source .workflow_venvs/.venv_benchmarks_run_script/bin/activate

# Send a test prompt
python3 utils/prompt_client_cli.py \
  --service-url http://localhost:8000/v1/completions \
  --prompt "What is the capital of France?" \
  --max-tokens 100
```

Or use curl:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "What is the capital of France?",
    "max_tokens": 100
  }'
```

## Optional Manual Model Setup

For manual control over model weights:

```bash
./setup.sh Llama-3.2-1B-Instruct
```

This script will:
1. Check system requirements (disk space, RAM)
2. Prompt for HF_TOKEN or Meta authorization
3. Download model weights
4. Store in `persistent_volume/` directory

## Advanced Options

### Development Mode
Mount local code changes into Docker container:

```bash
python3 run.py \
  --model Llama-3.2-1B-Instruct \
  --device n150 \
  --workflow server \
  --docker-server \
  --dev-mode
```

### Custom Port

```bash
python3 run.py \
  --model Llama-3.2-1B-Instruct \
  --device n150 \
  --workflow server \
  --docker-server \
  --service-port 9000
```

### Interactive Mode (for debugging)

```bash
python3 run.py \
  --model Llama-3.2-1B-Instruct \
  --device n150 \
  --workflow server \
  --docker-server \
  -it
```

### Multi-Device (T3K example)

```bash
python3 run.py \
  --model Llama-3.1-70B-Instruct \
  --device t3k \
  --workflow server \
  --docker-server \
  --device-id 0,1,2,3,4,5,6,7
```

### Skip Trace Capture (faster if already done)

```bash
python3 run.py \
  --model Llama-3.2-1B-Instruct \
  --device n150 \
  --workflow benchmarks \
  --docker-server \
  --disable-trace-capture
```

## Logs and Monitoring

All logs are in `workflow_logs/`:

```
workflow_logs/
├── run_logs/           # Main execution logs
├── docker_server/      # Docker container logs
├── benchmarks_output/  # Benchmark JSON results
├── evals_output/       # Evaluation results
└── reports_output/     # Summary reports (markdown)
```

View specific log:
```bash
# Main run log
cat workflow_logs/run_logs/run_<timestamp>_<model>_<device>_<workflow>.log

# Docker server log
docker logs -f <container-id>
```

## Troubleshooting

### 1. Python venv issues
```bash
python3 run.py --reset-venvs
```

### 2. Docker permission errors
```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 3. HuggingFace 401 Unauthorized
- Visit the model page (e.g., https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- Click "Access repository" and accept terms
- Update token in `.env` file if needed

### 4. Out of disk space
Check model requirements:
- Llama-3.2-1B: 4GB minimum
- Llama-3.1-8B: 32GB minimum
- Llama-3.1-70B: 350GB minimum

### 5. No Tenstorrent devices found
```bash
# Check devices
tt-smi -s

# Check topology (for multi-device)
tt-topology
```

## First Time Setup Summary

1. **Install prerequisites**
   ```bash
   apt install python3-venv
   ```

2. **Get HuggingFace token**
   - Visit https://huggingface.co/settings/tokens
   - Create a read token
   - Accept terms for your model repository

3. **Run first example**
   ```bash
   python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow server --docker-server
   ```

4. **Test the server**
   ```bash
   source .workflow_venvs/.venv_benchmarks_run_script/bin/activate
   python3 utils/prompt_client_cli.py \
     --service-url http://localhost:8000/v1/completions \
     --prompt "Hello, world!" \
     --max-tokens 50
   ```

## What Happens on First Run?

1. **Virtual Environment Setup** (~5-15 minutes)
   - `run.py` uses `uv` and `venv` to bootstrap environments
   - Creates `.workflow_venvs/` directory
   - Installs required dependencies

2. **Secrets Prompt**
   - HF_TOKEN for model access
   - JWT_SECRET for API authentication
   - Saved to `.env` file for future runs

3. **Model Setup** (via `setup_host.py`)
   - Downloads model weights from HuggingFace
   - Stores in `persistent_volume/` directory
   - Creates `persistent_volume/model_envs/<model>.env`

4. **Docker Pull**
   - Pulls pre-built Docker image from GHCR
   - Contains tt-metal and vLLM builds

5. **Server Start**
   - Mounts model weights and caches
   - Starts vLLM server in container
   - Loads model to Tenstorrent device

## Next Steps

- Explore different models and devices
- Run benchmarks to measure performance
- Run evals to check accuracy
- Check [docs/workflows_user_guide.md](docs/workflows_user_guide.md) for details
- See [docs/development.md](docs/development.md) for contributing

## Resources

- **Main README**: [README.md](README.md)
- **Workflows Guide**: [docs/workflows_user_guide.md](docs/workflows_user_guide.md)
- **Development Guide**: [docs/development.md](docs/development.md)
- **Benchmarking**: [benchmarking/README.md](benchmarking/README.md)
- **Evaluations**: [evals/README.md](evals/README.md)
- **GitHub Issues**: https://github.com/tenstorrent/tt-inference-server/issues


