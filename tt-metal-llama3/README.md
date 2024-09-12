# TT Metalium Llama 3.1 70B Inference API

This implementation also supports Llama 3 70B and Llama 2 70B. 

# Table of Contents

- [Quick run](#quick-run)
  - [Environment variables](#environment-variables)
  - [Docker Run - llama3 inference server](#docker-run---llama3-inference-server)
  - [JWT_TOKEN Authorization](#jwt_token-authorization)
  - [Docker Run - Interactive shell for llama3 demo scripts](#docker-run---interactive-shell-for-llama3-demo-scripts)
- [First run setup](#first-run-setup)
  - [1. Docker install](#1-docker-install)
  - [2. Ensure system dependencies installed](#2-ensure-system-dependencies-installed)
  - [3. CPU performance setting](#3-cpu-performance-setting)
  - [4. Docker image](#4-docker-image)
  - [5. Download weights](#5-download-weights)
    - [Download Llama 3.1 70B weights](#download-llama-31-70b-weights)
  - [6. Move and repack weights](#6-move-and-repack-weights)
    - [Copy over weights, tokenizer, params](#copy-over-weights-tokenizer-params)
    - [Repack the weights](#repack-the-weights)
- [Additional Documentation](#additional-documentation)

## Quick run

If first run setup has already been completed, start here. If first run setup has not been run please see the instructions below for [First run setup](#first-run-setup).

### Environment variables

```bash
cd tt-inference-server
cp .env.default .env
# set JWT_SECRET to a strong secret for production environments
vim .env
# .env is ignored from git
```

### Docker Run - llama3 inference server

Container will run `gunicorn --config gunicorn.conf.py` and start the inference server and model backend.
```bash
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama-3.1-8b-instructv0.0.1
docker run \
  --rm \
  -it \
  --env-file tt-metal-llama3/.env \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${PERSISTENT_VOLUME?ERROR env var PERSISTENT_VOLUME must be set}:/home/user/cache_root:rw \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-rc31-ba7c8de54023
```

The inference API server takes 3-5 minutes to start up (~60 minutes on first run when generating caches) then will start serving requests. To send HTTP requests to the inference server run the example scripts in a separate bash shell. You can use `docker exec -it <container-id> bash` to create a shell in the docker container or run the client scripts on the host ensuring the correct port mappings and python dependencies are available:
```bash
# see JWT_TOKEN Authorization section below
export AUTHORIZATION=<your token>
# send a single batch of requests
python test_inference_api.py
# run through 25 batches of 32 prompts from alpaca eval (800 total)
python test_inference_api_alpaca_eval.py
```

Example request to the inference-server using curl:
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
export JWT_ENCODED=$(python scripts/jwt_util.py --secret ${JWT_SECRET?ERROR env var JWT_SECRET must be set} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}')
export AUTHORIZATION="Bearer ${JWT_ENCODED}"
```



### Docker Run - Interactive shell for llama3 demo scripts

These demos show direct usage of the model implementation.

Run container overriding the entrypoint `CMD` with an interactive bash shell:
```bash
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
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
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-rc31-ba7c8de54023 bash
```

Within the container shell:
```bash
# run demo with pytest for llama3
cd /tt-metal
pytest -svv models/demos/t3000/llama3_70b/demo/demo.py::test_LlamaModel_demo[wormhole_b0-True-short_context-check_disabled-sampling-tt-70b-T3000-80L-prefill_decode-text_completion-llama3]

# this script will run through 800 samples of alpaca eval (25 batches of 32 users).
# outputs are appended to /home/user/cache_root/demo_user_output_{timestamp}.txt
python scripts/demo_llama3_alpaca_eval.py
```

## First run setup

### 1. Docker install

see Ubuntu apt guide: https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

and postinstall guide, to allow $USER to run docker without sudo: https://docs.docker.com/engine/install/linux-postinstall/

### 2. Ensure system dependencies installed

- tt-smi: https://github.com/tenstorrent/tt-smi
- firmware: bundle 80.10.1.0 (https://github.com/tenstorrent/tt-firmware/blob/02b4b6ed49b6ea2fb9a8664e99d4fed25e443bd6/experiments/fw_pack-80.10.1.0.fwbundle)
- drivers: tt-kmd version 1.28 (https://github.com/tenstorrent/tt-kmd/tree/ttkmd-1.28)
- topology: ensure mesh topology https://github.com/tenstorrent/tt-topology
- hugepages: see hugepages-setup.sh https://github.com/tenstorrent/tt-system-tools

### 3. CPU performance setting

```bash
sudo apt-get update && sudo apt-get install -y linux-tools-generic
# enable perf mode
sudo cpupower frequency-set -g performance
# disable perf mode
sudo cpupower frequency-set -g ondemand
```

### 4. Docker image

```bash
# pull image from GHCR
docker pull ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-rc31-ba7c8de54023
```

### 5. download weights
To download the Llama 3.1 weights from Meta (https://llama.meta.com/llama-downloads/), you will need to submit your contact email and company information to get the license URL for downloading.

Once you have the signed URL you can run the download scripts for the respective version.

#### Download Llama 3.1 70B weights

```bash
export LLAMA3_1_REPO=~/llama-models
git clone https://github.com/meta-llama/llama-models.git ${LLAMA3_1_REPO}
export LLAMA3_1_DIR=${LLAMA3_1_REPO}/models/llama3_1
cd ${LLAMA3_1_DIR}
./download.sh
# input meta-llama-3.1-70b
# then, input meta-llama-3.1-70b-instruct
```

NOTE: The input to ./download.sh for instruct fine-tuned models is different, you must first input `meta-llama-3.1-70b`, then input `meta-llama-3.1-70b-instruct` :
```log
 **** Model list ***
 -  meta-llama-3.1-405b
 -  meta-llama-3.1-70b
 -  meta-llama-3.1-8b
 -  meta-llama-guard-3-8b
 -  prompt-guard
Choose the model to download: meta-llama-3.1-70b

 Selected model: meta-llama-3.1-70b
 
 **** Available models to download: ***
 -  meta-llama-3.1-70b-instruct
 -  meta-llama-3.1-70b
Enter the list of models to download without spaces or press Enter for all: meta-llama-3.1-70b-instruct

<download progress>

Checking checksums
consolidated.00.pth: OK
consolidated.01.pth: OK
consolidated.02.pth: OK
consolidated.03.pth: OK
consolidated.04.pth: OK
consolidated.05.pth: OK
consolidated.06.pth: OK
consolidated.07.pth: OK
params.json: OK
tokenizer.model: OK
```

### 6. move and repack weights

Create a ubuntu user group for shared file access with container and host:
```bash
# group dockermount is used to share acces to users on HOST
sudo groupadd dockermount
sudo usermod -aG dockermount <host user>
# UID 1000 is the container user
# check which username has UID 1000
getent passwd 1000
sudo usermod -aG dockermount <UID 1000 user>
# refresh groups in current shell, may need to logout and back in to have on new shells
newgrp dockermount
```

The persistent volume root will be where each model weights, caches, and logs will be stored.
```bash
cd tt-inference-server
export PERSISTENT_VOLUME_ROOT=$PWD/persistent_volume/
mkdir -p ${PERSISTENT_VOLUME_ROOT}
```

#### copy over weights, tokenizer, params

```bash
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama3.1-70bv0.0.1
# create directories in persistent volume
mkdir -p ${PERSISTENT_VOLUME}/model_weights/repacked-llama-3.1-70b-instruct
mkdir -p ${PERSISTENT_VOLUME}/tt_metal_cache/cache_repacked-llama-3.1-70b-instruct
# assuming weights are downloaded to: ~/llama3/Meta-Llama-3-70B-Instruct/
cp -r ${LLAMA3_1_DIR}/Meta-Llama-3.1-70B-Instruct ${PERSISTENT_VOLUME}/model_weights/llama-3.1-70b-instruct
# copy tokenizer and params to repacked
cp ${LLAMA3_1_DIR}/Meta-Llama-3.1-70B-Instruct/tokenizer.model ${PERSISTENT_VOLUME}/model_weights/repacked-llama-3.1-70b-instruct/tokenizer.model
cp ${LLAMA3_1_DIR}/Meta-Llama-3.1-70B-Instruct/params.json ${PERSISTENT_VOLUME}/model_weights/repacked-llama-3.1-70b-instruct/params.json
# this will make the files readable inside and outside the container via group permissions
# UID 1000 is the container user
sudo chown -R 1000:dockermount ${PERSISTENT_VOLUME}
# set owner and group to r/w/x
sudo chmod -R 775 ${PERSISTENT_VOLUME}
```

#### Repack the weights

Use the docker container to run the `repack_weights.py` script. See section on how to use `.env.default` to add correct env vars in [Environment Variables](#environment-variables).
```bash
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
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-rc31-ba7c8de54023 bash

cd /tt-metal
python models/demos/t3000/llama2_70b/scripts/repack_weights.py /home/user/cache_root/model_weights/llama-3.1-70b-instruct ${LLAMA3_CKPT_DIR} 5
```

# Additional Documentation

- [FAQ](docs/faq.md)
- [Development](docs/development.md)
