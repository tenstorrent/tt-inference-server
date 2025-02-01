# vLLM TT Metalium Llama 3.3 70B Inference API

This implementation supports Llama 3.1 70B with vLLM at https://github.com/tenstorrent/vllm/tree/dev

## Table of Contents

- [Quick run](#quick-run)
  - [Docker Run - vLLM llama3 inference server](#docker-run---vllm-llama3-inference-server)
- [First run setup](#first-run-setup)
  - [1. Docker install](#1-docker-install)
  - [2. Ensure system dependencies installed](#2-ensure-system-dependencies-installed)
  - [3. CPU performance setting](#3-cpu-performance-setting)
  - [4. Docker image](#4-docker-image)
  - [5. Automated Setup: environment variables and weights files](#5-automated-setup-environment-variables-and-weights-files)
- [Additional Documentation](#additional-documentation)

## Quick run

If first run setup has already been completed, start here. If first run setup has not been run please see the instructions below for [First run setup](#first-run-setup).

### Docker Run - vLLM llama3 inference server

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
  ghcr.io/tenstorrent/tt-inference-server/vllm-llama3-src-dev-ubuntu-20.04-amd64:v0.0.1-47fb1a2fb6e0-2f33504bad49
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

You can use `docker exec -it <container-id> bash` to create a shell in the docker container or run the client scripts on the host (ensuring the correct port mappings and python dependencies):

#### Run example clients from within Docker container:
```bash
# oneliner to enter interactive shell on most recently ran container
docker exec -it $(docker ps -q | head -n1) bash

# inside interactive shell, run example clients script making requests to vLLM server:
cd ~/app/src
# this example runs a single request from alpaca eval, expecting and parsing the streaming response
python example_requests_client_alpaca_eval.py --stream True --n_samples 1 --num_full_iterations 1 --batch_size 1
# this example runs a full-dataset stress test with 32 simultaneous users making requests
python example_requests_client_alpaca_eval.py --stream True --n_samples 805 --num_full_iterations 1 --batch_size 32
```

## First run setup

Tested starting condition is from a fresh installation of Ubuntu 20.04 with Tenstorrent system dependencies installed.

### 1. Docker install

see Ubuntu apt guide: https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

Recommended to follow postinstall guide to allow $USER to run docker without sudo: https://docs.docker.com/engine/install/linux-postinstall/

### 2. Ensure system dependencies installed

Follow TT strating guide software installation at: https://docs.tenstorrent.com/quickstart.html

Ensure all set up:
- firmware: tt-firmware (https://github.com/tenstorrent/tt-firmware)
- drivers: tt-kmd (https://github.com/tenstorrent/tt-kmd)
- hugepages: see https://docs.tenstorrent.com/quickstart.html#step-4-setup-hugepages and https://github.com/tenstorrent/tt-system-tools
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
docker pull ghcr.io/tenstorrent/tt-inference-server/vllm-llama3-src-dev-ubuntu-20.04-amd64:v0.0.1-47fb1a2fb6e0-2f33504bad49
```

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
./setup.sh Llama-3.3-70B-instruct
```

# Additional Documentation

- [Development](docs/development.md)
- [Benchmarking](../benchmarking/README.md)
- [Evals](../evals/README.md)
- [Locust load testsing](../locust/README.md)
- [tests](../tests/README.md)
