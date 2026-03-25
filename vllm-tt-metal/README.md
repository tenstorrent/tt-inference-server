# vLLM tt-metal Docker image

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

See [docs/prerequisites.md](../docs/prerequisites.md)

### Run model in Docker container

There are two ways to run the inference server in Docker:

For `run.py` usage see [docs/workflows_user_guide.md](../docs/workflows_user_guide.md)

See the full [run.py CLI documentation](../workflows/README.md#runpy-cli-usage) for all options including Docker volume strategies and `--print-docker-cmd`.

The container can be used independently from `run.py`. See the [Container Interface](#container-interface-direct-docker-run) section below.
The inference server container can be run directly with `docker run`, without `run.py`. The container entrypoint (`run_vllm_api_server.py`) accepts `--model` and `--tt-device` to resolve the model configuration from a bundled model spec JSON.

### Minimal Example

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_tt_transformers-Llama-3.2-1B-Instruct:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-84b4c53-222ee06 \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --tt-device n300
```

Required Docker flags:
- `--device /dev/tenstorrent` -- passes the Tenstorrent device into the container
- `--ipc host` -- allows for shared memory with host
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

**File permissions:** The container runs as a non-root user (`container_app_user`). There is no root-level entrypoint that adjusts permissions at startup, so mounted volumes must already be accessible to the image's built-in UID (UID `1000` for default release images).

| Strategy | Host permission requirement |
|---|---|
| Docker named volume (default) | None. Docker seeds the volume from the image with correct ownership on first creation. |
| Host persistent volume (bind mount) | The host directory must be **writable** by the image UID (e.g. `sudo chown 1000 <path>`). If needed add Docker user UID (e.g. `1000`) to a shared group with user who owns the directory, then `chown` to the shared group. With this, `chmod` to `775` For read+write. Alternatively, if that does not work for your usecase you can just do `chmod` to `777` without the shared group. |
| Host weights / Host HF cache (readonly bind mounts) | The host path only needs to be **readable** by the image UID. TT Metal caches are stored in a separate Docker named volume, not on the host. |

**1. Host persistent volume**

Bind mount an entire host directory as the container's `cache_root`. All data (weights, TT Metal caches) lives on the host filesystem. The host directory must be writable by the image's built-in UID (see permissions table above). This is equivalent to `run.py --host-volume`.

```bash
docker run \
  ... \
  --mount type=bind,src=$HOST_VOLUME,dst=/home/container_app_user/cache_root \
  <image> --model <model> --tt-device <device>
```

**2. Host model weights directory**

Mount a host directory containing pre-downloaded model weights readonly. TT Metal caches use a separate Docker named volume. This is equivalent to `run.py --host-weights-dir`.

```bash
docker run \
  ... \
  --mount type=bind,src=$HOST_WEIGHTS_DIR,dst=/home/container_app_user/readonly_weights_mount/<dir-name>,readonly \
  -e MODEL_WEIGHTS_DIR=/home/container_app_user/readonly_weights_mount/<dir-name> \
  --volume <volume-name>:/home/container_app_user/cache_root \
  <image> --model <model> --tt-device <device>
```

This can also use the host's existing HuggingFace cache. When using `run.py --host-hf-cache`, pass the top-level HuggingFace cache directory (typically `~/.cache/huggingface`). `run.py` automatically resolves the snapshot path inside the cache, mounts the model repo directory, and sets `MODEL_WEIGHTS_DIR` to point at the correct snapshot inside the container.

When using `docker run` directly, you must resolve the snapshot path yourself and pass it as the bind-mount source. The HF cache layout is typically like:
```
~/.cache/huggingface/hub/models--<org>--<model-name>/snapshots/<revision-hash>/
```

Set `$HF_CACHE_SNAPSHOT` to that resolved path, for example:

```bash
HF_CACHE_SNAPSHOT=~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/<revision-hash>
```

```bash
docker run \
  ... \
  --mount type=bind,src=$HF_CACHE_SNAPSHOT,dst=/home/container_app_user/readonly_weights_mount/<model-name>,readonly \
  -e MODEL_WEIGHTS_DIR=/home/container_app_user/readonly_weights_mount/<model-name> \
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

# for example HTTP request using curl, assuming SERVICE_PORT=8000
export API_URL="http://0.0.0.0:8000/v1/chat/completions"
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
