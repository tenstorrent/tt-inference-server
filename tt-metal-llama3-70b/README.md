# TT Metalium Llama 3.1 70B Inference API

This implementation supports Llama 3.1 70B, Llama 3 70B, and Llama 2 70B. 

## Table of Contents

- [Quick run](#quick-run)
  - [Docker Run - llama3 inference server](#docker-run---llama3-inference-server)
  - [JWT_TOKEN Authorization](#jwt_token-authorization)
  - [Docker Run - Interactive shell for llama3 demo scripts](#docker-run---interactive-shell-for-llama3-demo-scripts)
- [First run setup](#first-run-setup)
  - [1. Docker install](#1-docker-install)
  - [2. Ensure system dependencies installed](#2-ensure-system-dependencies-installed)
  - [3. CPU performance setting](#3-cpu-performance-setting)
  - [4. Docker image](#4-docker-image)
  - [5. Automated Setup: environment variables and weights files](#5-automated-setup-environment-variables-and-weights-files)
- [Additional Documentation](#additional-documentation)

## Quick run

If first run setup has already been completed, start here. If first run setup has not been run please see the instructions below for [First run setup](#first-run-setup).


### Docker Run - llama3 inference server

Container will run `gunicorn --config gunicorn.conf.py` and start the inference server and model backend.
```bash
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama-3.1-70b-instructv0.0.1/
docker run \
  --rm \
  -it \
  --env-file tt-metal-llama3-70b/.env \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${PERSISTENT_VOLUME?ERROR env var PERSISTENT_VOLUME must be set}:/home/user/cache_root:rw \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.52.0-rc31-9d3be887987b
```

The inference API server takes 3-5 minutes to start up (~60 minutes on first run when generating caches) then will start serving requests. To send HTTP requests to the inference server run the example scripts in a separate bash shell. You can use `docker exec -it <container-id> bash` to create a shell in the docker container or run the client scripts on the host ensuring the correct port mappings and python dependencies are available:
```bash
# see JWT_TOKEN Authorization section below for details
export AUTHORIZATION="Bearer $(python scripts/jwt_util.py --secret ${JWT_SECRET?ERROR env var JWT_SECRET must be set} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}')"
# send a single batch of requests
python test_inference_api.py
# run through 25 batches of 32 prompts from alpaca eval (800 total)
python test_inference_api_alpaca_eval.py
```

Example request to the inference-server using curl:
Note: curl will only print the output of text streaming after a newline is sent, or the response completes.
```bash
curl -X POST http://127.0.0.1:7000/inference/llama3-70b \
-H "Authorization: $AUTHORIZATION" \
-H "Content-Type: application/json" \
-d '{
    "text": "What is in Austin Texas?",
    "max_tokens": 128
}'
```

To stop the container, use `docker stop <container-id>`. A soft reset may be required.

### JWT_TOKEN Authorization

To authenticate requests use the header `Authorization`. The JWT token can be computed using the script `jwt_util.py`. This is an example:
```bash
export JWT_SECRET=<your-secure-secret>
export AUTHORIZATION="Bearer $(python scripts/jwt_util.py --secret ${JWT_SECRET?ERROR env var JWT_SECRET must be set} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}')"
```

### Docker Run - Interactive shell for llama3 demo scripts

These demos show direct usage of the model implementation.

Run container overriding the entrypoint `CMD` with an interactive bash shell:
```bash
cd tt-inference-server
# Ensure that if you have already set up the model weights and cache, you are using the correct persistent volume.
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama3.1-70bv0.0.1
docker run \
  --rm \
  -it \
  --env-file tt-metal-llama3-70b/.env \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${PERSISTENT_VOLUME?ERROR env var PERSISTENT_VOLUME must be set}:/home/user/cache_root:rw \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.52.0-rc31-9d3be887987b bash
```

Within the container shell:
```bash
# run demo with pytest for llama3
cd /tt-metal
pytest -svv models/demos/t3000/llama2_70b/demo/demo_continuous_batching.py::test_LlamaModel_demo[wormhole_b0-True-short_context-greedy-tt-70b-T3000-80L-prefill_decode-chat_completion-llama3]

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

```bash
# pull image from GHCR
docker pull ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.52.0-rc31-9d3be887987b
```

### 5. Automated Setup: environment variables and weights files

The script `tt-metal-llama3-70b/setup.sh` automates:

1. interactively creating the .env file,
2. downloading the Llama model weights,
3. repacking the weights as required for tt-metal implementation,
4. creating the default persistent storage directory structure and permissions.

```bash
cd tt-inference-server/tt-metal-llama3-70b
chmod +x setup.sh
./setup.sh llama-3.1-70b-instruct
```

NOTE: for instruct fine-tuned models, you must first input `llama-3.1-70b-instruct`, then when running `download.sh`, input `meta-llama-3.1-70b`, and finally input `meta-llama-3.1-70b-instruct`.

If you need to modify the setup or otherwise need to manually do it please see the [Manual Setup Guide](docs/manual_setup_guide.md).

# Additional Documentation

- [FAQ](docs/faq.md)
- [Development](docs/development.md)
