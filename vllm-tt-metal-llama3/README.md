# vLLM TT Metalium Llama 3.1 70B Inference API

This implementation supports Llama 3.1 70B with vLLM at https://github.com/tenstorrent/vllm/tree/dev

## Table of Contents

- [Quick run](#quick-run)
  - [Docker Run - vLLM llama3 inference server](#docker-run---llama3-inference-server)
(#docker-run---interactive-shell-for-llama3-demo-scripts)
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
export MODEL_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama-3.1-70b-instructv0.0.1/
docker run \
  --rm \
  -it \
  --env-file persistent_volume/model_envs/llama-3.1-70b-instruct.env \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${MODEL_VOLUME?ERROR env var MODEL_VOLUME must be set}:/home/user/cache_root:rw \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm:v0.0.1-tt-metal-v0.54.0-rc2-953161188c50 
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

- tt-smi: https://github.com/tenstorrent/tt-smi
- firmware: bundle 80.10.1.0 (https://github.com/tenstorrent/tt-firmware/blob/02b4b6ed49b6ea2fb9a8664e99d4fed25e443bd6/experiments/fw_pack-80.10.1.0.fwbundle)
- drivers: tt-kmd version 1.29 (https://github.com/tenstorrent/tt-kmd/tree/ttkmd-1.29)
- topology: ensure mesh topology https://github.com/tenstorrent/tt-topology
- hugepages: https://github.com/tenstorrent/tt-system-tools

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
docker pull ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm:v0.0.1-tt-metal-v0.54.0-rc2-953161188c50
```

#### Option B: Build Docker Image

For instructions on building the Docker imagem locally see: [vllm-tt-metal-llama3/docs/development](../vllm-tt-metal-llama3/docs/development.md#step-1-build-docker-image)

### 5. Automated Setup: environment variables and weights files

The script `setup.sh` automates:

1. interactively creating the .env file,
2. downloading the Llama model weights,
3. repacking the weights as required for tt-metal implementation,
4. creating the default persistent storage directory structure and permissions.

```bash
cd tt-inference-server
chmod +x setup.sh
./setup.sh llama-3.1-70b-instruct
```

NOTE: for instruct fine-tuned models, you must first input `llama-3.1-70b-instruct`, then when running `download.sh`, input `meta-llama-3.1-70b`, and finally input `meta-llama-3.1-70b-instruct`.

If you need to modify the setup or otherwise need to manually do it please see the [Manual Setup Guide](docs/manual_setup_guide.md).

# Additional Documentation

- [FAQ](docs/faq.md)
- [Development](docs/development.md)
