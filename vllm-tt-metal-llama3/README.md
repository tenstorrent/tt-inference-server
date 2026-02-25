# vLLM TT Metalium TT-Transformer Inference API

This implementation supports the following models in the [LLM model list](../README.md#llms)

## Table of Contents

- [Setup and Installation](#setup-and-installation)
  - [1. Docker Install](#1-docker-install)
  - [2. Ensure System Dependencies Installed](#2-ensure-system-dependencies-installed)
  - [3. CPU Performance Setting](#3-cpu-performance-setting)
  - [4. Run Model in Docker](#4-run-model-in-docker)
- [Container Interface (Direct Docker Run)](#container-interface-direct-docker-run)
  - [Container CLI Arguments](#container-cli-arguments)
  - [Secrets](#secrets)
  - [Persistent Volume Overrides](#persistent-volume-overrides)
- [Example Clients](#example-clients)
  - [Run Example Clients from Within Docker Container](#run-example-clients-from-within-docker-container)

## Setup and installation

This guide was tested starting condition is from a fresh installation of Ubuntu 20.04 with Tenstorrent system dependencies installed. 
Ubuntu 22.04 should also work for most if not all models. 

### 1. Docker install

see Ubuntu apt guide: https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

Recommended to follow postinstall guide to allow $USER to run docker without sudo: https://docs.docker.com/engine/install/linux-postinstall/

### 2. Ensure system dependencies installed

Follow TT guide software installation at: https://docs.tenstorrent.com/getting-started/README

Ensure all set up:
- firmware: tt-firmware (https://github.com/tenstorrent/tt-firmware)
- drivers: tt-kmd (https://github.com/tenstorrent/tt-kmd)
- hugepages: see https://docs.tenstorrent.com/getting-started/README#step-4-set-up-hugepages
- tt-smi: https://github.com/tenstorrent/tt-smi

If running on a TT-LoudBox or TT-QuietBox, you will also need:
- topology: tt-topology https://github.com/tenstorrent/tt-topology
  - set up mesh topology, see https://github.com/tenstorrent/tt-topology?tab=readme-ov-file#mesh

### 3. CPU performance setting

In order to get peak performance increasing the CPU frequency profile is recommended. If you cannot do this for your setup, it is optional and can be skipped, though performance may be lower than otherwise expected.

```bash
sudo apt-get update && sudo apt-get install -y linux-tools-generic
# enable perf mode
sudo cpupower frequency-set -g performance

# disable perf mode (if desired later)
# sudo cpupower frequency-set -g ondemand
```

### 4. Run model in Docker

There are two ways to run the inference server in Docker:

**Option A: Using `run.py` automation (recommended for first-time setup)**

`run.py` handles model weights download, Docker image pull, host setup, and secret management automatically. Model weights download, Docker image download, and Python environment setup may take a long time on low-bandwidth networks.

```bash
python3 run.py --model Llama-3.2-1B-Instruct --tt-device n300 --workflow server --docker-server
```

See the full [run.py CLI documentation](../workflows/README.md#runpy-cli-usage) for all options including Docker volume strategies and `--print-docker-cmd`.

**Option B: Direct `docker run` (container interface)**

The container can be used independently from `run.py`. See the [Container Interface](#container-interface-direct-docker-run) section below.

## Container Interface (Direct Docker Run)

The inference server container can be run directly with `docker run`, without `run.py`. The container entrypoint (`run_vllm_api_server.py`) accepts `--model` and `--tt-device` to resolve the model configuration from a bundled model spec JSON.

### Minimal Example

```bash
docker run \
  --rm \
  --device /dev/tenstorrent \
  --cap-add SYS_NICE \
  --shm-size 32G \
  --publish 8000:8000 \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_tt_transformers-Llama-3.1-8B-Instruct:/home/container_app_user/cache_root \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tt-device n300
```

Required Docker flags:
- `--device /dev/tenstorrent` -- passes the Tenstorrent device into the container
- `--cap-add SYS_NICE` -- required for thread priority management
- `--shm-size 32G` -- shared memory for model loading
- `--mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G` -- hugepages for TT Metal

### Container CLI Arguments

These arguments are passed after the Docker image name:

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model` | Yes | -- | HuggingFace model repo (e.g., `meta-llama/Llama-3.1-8B-Instruct`). |
| `--tt-device` | Yes | -- | Device type: `n150`, `n300`, `t3k`, `galaxy`, etc. |
| `--engine` | No | Model spec default | Inference engine override: `vllm`, `media`, `forge`. |
| `--impl` | No | Model spec default | Implementation name override (e.g., `tt-transformers`). |

### Secrets

Secrets are passed to the container via Docker environment variables or an env file:

```bash
# Option 1: env file
docker run ... --env-file ./.env ...

# Option 2: individual environment variables
docker run ... -e HF_TOKEN=$HF_TOKEN -e JWT_SECRET=$JWT_SECRET ...
```

- `HF_TOKEN` -- required for downloading model weights from HuggingFace (gated models).
- `JWT_SECRET` -- optional. When set, HTTP requests to the vLLM API require a bearer token in the `Authorization` header.

### Persistent Volume Overrides

By default, the container downloads model weights and stores TT Metal caches in a Docker named volume mounted at `/home/container_app_user/cache_root`. The following overrides change how persistent data is managed.

Only one strategy should be used at a time.

**1. Host persistent volume**

Bind mount an entire host directory as the container's `cache_root`. All data (weights, TT Metal caches) lives on the host filesystem. This is equivalent to `run.py --host-volume`.

```bash
docker run \
  ... \
  --mount type=bind,src=$HOST_VOLUME,dst=/home/container_app_user/cache_root \
  -e PERSISTENT_VOLUME=/home/container_app_user/cache_root \
  <image> --model <model> --tt-device <device>
```

**2. Host HuggingFace cache**

Mount the host's existing HuggingFace cache readonly. The container uses the mounted weights instead of downloading them. Other persistent data (TT Metal caches) uses a Docker named volume. This is equivalent to `run.py --host-hf-cache`.

```bash
docker run \
  ... \
  --mount type=bind,src=$HF_CACHE,dst=/home/container_app_user/readonly_weights_mount/<model-name>,readonly \
  -e MODEL_WEIGHTS_DIR=/home/container_app_user/readonly_weights_mount/<model-name> \
  --volume <volume-name>:/home/container_app_user/cache_root \
  <image> --model <model> --tt-device <device>
```

**3. Host model weights directory**

Mount a host directory containing pre-downloaded model weights readonly. Other persistent data uses a Docker named volume. This is equivalent to `run.py --host-weights-dir`.

```bash
docker run \
  ... \
  --mount type=bind,src=$HOST_WEIGHTS_DIR,dst=/home/container_app_user/readonly_weights_mount/<dir-name>,readonly \
  -e MODEL_WEIGHTS_DIR=/home/container_app_user/readonly_weights_mount/<dir-name> \
  --volume <volume-name>:/home/container_app_user/cache_root \
  <image> --model <model> --tt-device <device>
```

### Example clients

Aside for [automated workflows](../docs/workflows_user_guide.md) to send prompts to the inference server, you can send prompts directly to the model following these steps.

Use `docker exec --user 1000 -it <container-id> bash` (--user uid must match container you are using, default is 1000) to create a shell in the docker container or run the client scripts on the host (ensuring the correct port mappings and python dependencies):

#### Run example clients from within Docker container:
```bash
# oneliner to enter interactive shell on most recently ran container
docker exec -it $(docker ps -q | head -n1) bash

# inside interactive shell, run example clients script to send prompt request to vLLM server:
cd ~/app/src
python example_requests_client.py
```

### vLLM API Server Authorization

If `JWT_SECRET` is set, HTTP requests to vLLM API require bearer token in 'Authorization' header. See docs for how to get bearer token.

Setting JWT authorization is optional, if unset the server will not require the 'Authorization' header to be set and will not check it for a JWT match.

```bash
export JWT_SECRET="my-secret-string"
export BEARER_TOKEN=$(python -c 'import os, json, jwt; print(jwt.encode({"team_id": "tenstorrent", "token_id": "debug-test"}, os.getenv("JWT_SECRET"), algorithm="HS256"))')

# for example HTTP request using curl, assuming SERVICE_PORT=7000
export API_URL="http://0.0.0.0:7000/v1/chat/completions"
curl -s --no-buffer -X POST "${API_URL}" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $BEARER_TOKEN" \
    -d '{
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "messages": [
        {
            "role": "user",
            "content": "What is Tenstorrent?"
        }
        ],
        "max_tokens": 256
    }'
```
