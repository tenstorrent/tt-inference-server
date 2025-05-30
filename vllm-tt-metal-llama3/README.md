# vLLM TT Metalium TT-Transformer Inference API

This implementation supports the following models in the [LLM model list](../README.md#llms)

## Table of Contents

- [Setup and Installation](#setup-and-installation)
  - [1. Docker Install](#1-docker-install)
  - [2. Ensure System Dependencies Installed](#2-ensure-system-dependencies-installed)
  - [3. CPU Performance Setting](#3-cpu-performance-setting)
  - [4. Run Model in Docker with `run.py` Automation](#4-run-model-in-docker-with-runpy-automation)
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

### 4. Run model in Docker with `run.py` automation

The `run.py` automation will setup the model

model weights download, docker image download, and Python environment may take a long time on low-bandwidth network.


```bash
# run inference server with docker
python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow server --docker-server
```

See documentation in [Model Readiness Workflows](../docs/workflows_user_guide.md#docker-server)

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
