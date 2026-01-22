# Manual Docker Run Setup

### ⚠️ **DEPRECATION WARNING:** not tested any more
Follow [Model Readiness Workflows](../../docs/workflows_user_guide.md) for supported automated usage.

### Docker image

Either download the Docker image from GitHub Container Registry (recommended for first run) or build the Docker image locally using the dockerfile.

#### Option A: GitHub Container Registry

```bash
# pull image from GHCR
docker pull ghcr.io/tenstorrent/tt-inference-server/vllm-llama3-src-dev-ubuntu-20.04-amd64:{VERSION}-{tt-metal-tag}-{vllm-commit-sha}
```

Note: as the docker image is downloading you can continue to the next step and download the model weights in parallel.

#### Option B: Build Docker Image

For instructions on building the Docker imagem locally see: [vllm-tt-metal/docs/development](../vllm-tt-metal/docs/development.md#step-1-build-docker-image)

### Weights Setup: environment variables and weights files

The script `setup.sh` automates:

1. interactively creating the model specific .env file,
2. downloading the model weights,
3. (if required) repacking the weights for tt-metal implementation,
4. creating the default persistent storage directory structure and permissions.

```bash
cd tt-inference-server
chmod +x setup.sh
./setup.sh Llama-3.3-70B-Instruct
```

## Quick run

If first run setup above has already been completed, start here. If first run setup has not been completed, complete [Setup and installation](#setup-and-installation).

### Docker Run - vLLM inference server

Run the container from the project root at `tt-inference-server`:
```bash
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export MODEL_NAME=Llama-3.3-70B-Instruct
export MODEL_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-${MODEL_NAME}-v0.0.1/
docker run \
  --rm \
  -it \
  --env-file persistent_volume/model_envs/${MODEL_NAME}.env \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${MODEL_VOLUME?ERROR env var MODEL_VOLUME must be set}:/home/container_app_user/cache_root:rw \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-20.04-amd64:{VERSION}-{tt-metal-tag}-{vllm-commit-sha}
```

By default the Docker container will start running the entrypoint command wrapped in `src/run_vllm_api_server.py`.
This can be run manually if you override the the container default command with an interactive shell via `bash`. 
In an interactive shell you can start the vLLM API server via:
```bash
# run server manually
python run_vllm_api_server.py
```

The vLLM inference API server takes 3-5 minutes to start up (~40-60 minutes on first run when generating caches) then will start serving requests. To send HTTP requests to the inference server run the example scripts in a separate bash shell.