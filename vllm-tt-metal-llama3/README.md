# vLLM TT Metalium TT-Transformer Inference API

This implementation supports the following models in the [LLM model list](../README.md#llms) with vLLM at https://github.com/tenstorrent/vllm/tree/dev

You can setup the model being deployed using the `setup.sh` script and `MODEL_NAME` environment variable to point to it as shown below. The examples below are using `MODEL_NAME=Llama-3.3-70B-Instruct`. It is recommended to use Instruct fine-tuned models for interactive use. Start with this if you're unsure.

## Table of Contents

- [Setup and installation](#setup-and-installation)
  - [1. Docker install](#1-docker-install)
  - [2. Ensure system dependencies installed](#2-ensure-system-dependencies-installed)
  - [3. CPU performance setting](#3-cpu-performance-setting)
  - [4. Docker image](#4-docker-image)
  - [5. Automated Setup: environment variables and weights files](#5-automated-setup-environment-variables-and-weights-files)
- [Quick run](#quick-run)
  - [Docker Run - vLLM inference server](#docker-run---vllm-inference-server)
- [Additional Documentation](#additional-documentation)

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

### 4. Docker image

Either download the Docker image from GitHub Container Registry (recommended for first run) or build the Docker image locally using the dockerfile.

#### Option A: GitHub Container Registry

```bash
# pull image from GHCR
docker pull ghcr.io/tenstorrent/tt-inference-server/vllm-llama3-src-dev-ubuntu-20.04-amd64:v0.0.1-b6ecf68e706b-b9564bf364e9
```

Note: as the docker image is downloading you can continue to the next step and download the model weights in parallel.

#### Option B: Build Docker Image

For instructions on building the Docker imagem locally see: [vllm-tt-metal-llama3/docs/development](../vllm-tt-metal-llama3/docs/development.md#step-1-build-docker-image)

### 5. Automated Setup: environment variables and weights files

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
  ghcr.io/tenstorrent/tt-inference-server/vllm-llama3-src-dev-ubuntu-20.04-amd64:v0.0.1-b6ecf68e706b-b9564bf364e9
```

By default the Docker container will start running the entrypoint command wrapped in `src/run_vllm_api_server.py`.
This can be run manually if you override the the container default command with an interactive shell via `bash`. 
In an interactive shell you can start the vLLM API server via:
```bash
# run server manually
python run_vllm_api_server.py
```

The vLLM inference API server takes 3-5 minutes to start up (~40-60 minutes on first run when generating caches) then will start serving requests. To send HTTP requests to the inference server run the example scripts in a separate bash shell. 

### Example clients

You can use `docker exec --user 1000 -it <container-id> bash` (--user uid must match container you are using, default is 1000) to create a shell in the docker container or run the client scripts on the host (ensuring the correct port mappings and python dependencies):

#### Run example clients from within Docker container:
```bash
# oneliner to enter interactive shell on most recently ran container
docker exec -it $(docker ps -q | head -n1) bash

# inside interactive shell, run example clients script to send prompt request to vLLM server:
cd ~/app/src
python example_requests_client.py
```

# Additional Documentation

- [Development](docs/development.md)
- [Benchmarking](../benchmarking/README.md)
- [Evals](../evals/README.md)
- [Locust load testsing](../locust/README.md)
- [tests](../tests/README.md)
