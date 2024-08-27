# TT Metalium Llama 3.1 70B Inference API

This implementation also supports Llama 3 70B and Llama 2 70B.

# Table of Contents

- [Quick run](#quick-run)
  - [Docker run - llama3 demo scripts](#docker-run---llama3-demo-scripts)
  - [Docker run - llama3 inference server](#docker-run---llama3-inference-server)
    - [llama 3 70B](#llama-3-70b)
    - [Llama 2 70B](#llama-2-70b)
  - [JWT_TOKEN Authorization](#jwt_token-authorization)
  - [Send requests using alpaca eval prompts](#send-requests-using-alpaca-eval-prompts)
- [Tenstorrent device soft resets](#tenstorrent-device-soft-resets)
- [First run setup](#first-run-setup)
  - [1. Docker install](#1-docker-install)
  - [2. Ensure tt-smi, firmware, drivers, and topology are correct](#2-ensure-tt-smi-firmware-drivers-and-topology-are-correct)
    - [tt-smi](#tt-smi)
    - [Firmware](#firmware)
    - [Drivers](#drivers)
    - [Topology](#topology)
    - [Hugepages setup](#hugepages-setup)
  - [3. CPU performance setting](#3-cpu-performance-setting)
  - [4. Docker image build](#4-docker-image-build)
  - [5. download weights](#5-download-weights)
    - [Download Llama 3.1 70B weights](#download-llama-31-70b-weights)
    - [Download Llama 3 70B weights](#download-llama-3-70b-weights)
  - [6. move and repack weights](#6-move-and-repack-weights)
    - [Llama 3.1 70B](#llama-31-70b)
    - [Llama 3 70B](#llama-3-70b)
    - [Llama 2 70B (skip if you only want to run Llama 3 70B)](#llama-2-70b-skip-if-you-only-want-to-run-llama-3-70b)
    - [Repack the weights](#repack-the-weights)
- [System dependencies](#system-dependencies)
- [Development](#development)
- [Run tests](#run-tests)
  - [Test with mocks](#test-with-mocks)
  - [Test with full on device backend](#test-with-full-on-device-backend)

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

### Docker run - llama3 demo scripts

These demos show direct usage of the model implementation for performance.

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
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-ba7c8de5 bash
```

To stop the container, simply exit the interactive shell.

Within the container shell:
```bash
# run demo with pytest for llama3
cd /tt-metal
pytest -svv models/demos/t3000/llama3_70b/demo/demo.py::test_LlamaModel_demo[wormhole_b0-True-short_context-check_disabled-sampling-tt-70b-T3000-80L-decode_only-text_completion-llama3]

# this script will run through 800 samples of alpaca eval (25 batches of 32 users).
# outputs are appended to /home/user/cache_root/demo_user_output_{timestamp}.txt
python tt_metal_impl/eval/demo_llama3_alpaca_eval.py
```

### Docker Run - llama3 inference server

Without overriding the entrypoint the container will run `gunicorn --config gunicorn.conf.py` and start the inference server and model backend.
WARNING: do not use insecure JWT_SECRET in an internet facing deployment, you must use a strong and secure secret for authentication.
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
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-ba7c8de5
```
Note: if you want to change the port, change SERVICE_PORT and the published port mapping (`--publish 7000:7000` above)

To send HTTP requests to the inference server that is run via `gunicorn --config gunicorn.conf.py` run the example scripts in a separate bash shell (you can use `docker exec -it <id> bash` to create a shell in the docker container):
```bash
# see JWT_TOKEN Authorization section below
export AUTHORIZATION=<your token>
# send a single batch of requests
python test_inference_api.py
# run through 25 batches of 32 prompts from alpaca eval (800 total)
python test_inference_api_alpaca_eval.py
```

##### llama 3 70B (optional)
```bash
# need to set path environment variables for demo scripts using different weights
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama3-70bv0.0.1
export LLAMA_VERSION=llama3
export LLAMA3_CKPT_DIR=/home/user/cache_root/model_weights/repacked-llama-3-70b-instruct
export LLAMA3_TOKENIZER_PATH=/home/user/cache_root/model_weights/repacked-llama-3-70b-instruct/tokenizer.model
export LLAMA3_CACHE_PATH=/home/user/cache_root/tt_metal_cache/cache_repacked-llama-3-70b-instruct
```

##### Llama 2 70B (optional)
```bash
# need to set path environment variables for demo scripts using different weights
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama2-70bv0.0.1
export LLAMA_VERSION=llama2
export LLAMA3_CKPT_DIR=/home/user/cache_root/model_weights/repacked-llama-2-70b-instruct
export LLAMA3_TOKENIZER_PATH=/home/user/cache_root/model_weights/repacked-llama-2-70b-instruct/tokenizer.model
export LLAMA3_CACHE_PATH=/home/user/cache_root/tt_metal_cache/cache_repacked-llama-2-70b-instruct
```

You can view the alpaca eval responses by copying the output file to the host, for example:
```bash
docker cp 3be74f228f5c:/home/user/cache_root/demo_user_output_2024-07-03_13-18-25.txt
```


To stop the container, use `docker stop $container_id`. A `tt-smi -r 0,1,2,3` reset will almost definitely be required as this will not shutdown the Tenstorrent devices gracefully.

The inference API server after start up (3-5 minutes) is available to server requests.
See the test scripts for examples on how to send those requests.

The requests can be sent from anywhere that can send HTTP requests to the published port mapped to internal SERVICE_PORT (7000 above). 

### JWT_TOKEN Authorization

To authenticate requests use the header `Authorization`. The JWT token can be computed using the script `jwt_util.py`. This is an example:
```bash
export JWT_SECRET=<your-secure-secret>
export JWT_ENCODED=$(python src/tt_metal_impl/scripts/jwt_util.py --secret ${JWT_SECRET?ERROR env var JWT_SECRET must be set} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}')
export AUTHORIZATION="Bearer ${JWT_ENCODED}"
```

The only dependency for this script is pyjwt:
```bash
pip install pyjwt==2.7.0
```

For example, without using the script above:
```python
import json
import jwt
jwt_secret = <your-secure-secret>
json_payload = json.loads('{"team_id": "tenstorrent", "token_id":"debug-test"}')
encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
print(encoded_jwt)
```

You can run this snippet or the `jwt_util.py` script with the JWT_SECRET in either:
1. the docker container interactive shell, or 
2. on the host in a python venv with pyjwt installed. (e.g. using the .venv mentioned below for running alpaca eval)

### Send requests using alpaca eval prompts

The `test_inference_api_alpaca_eval.py` script will run through 800 samples of alpaca eval (25 batches of 32 users).
The results are appended per batch to `responses_{datetime}.json`.

```bash
cd tt-metal-llama3-70b
# see above for JWT_TOKEN Authorization
export AUTHORIZATION="Bearer ${JWT_ENCODED}"
export CACHE_ROOT="test"  # just for testing on the host or external to container
# the huggingface datasets library is need to access alpaca eval
python3 -m venv .venv
source .venv/bin/activate
pip install datasets pyjwt==2.7.0
# run script
python src/test_inference_api_alpaca_eval.py
```

## Tenstorrent device soft resets

On host, use tt-smi (https://github.com/tenstorrent/tt-smi) to reset the n300 devices: 
```bash
# source
source ~/.venv/bin/activate
tt-smi -r 0,1,2,3
```

This soft reset is required for example when the device is not closed correctly during termination.
When this occurs the device may not be able to connect and train the ethernet links. If this occurs try soft resetting the device.

## First run setup

### 1. Docker install

see Ubuntu apt guide: https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository

and postinstall guide, to allow $USER to run docker without sudo: https://docs.docker.com/engine/install/linux-postinstall/

### 2. Ensure tt-smi, firmware, drivers, and topology are correct

#### tt-smi

see installation instructions in https://github.com/tenstorrent/tt-smi
```bash
git clone https://github.com/tenstorrent/tt-smi.git
cd tt-smi
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
source "$HOME/.cargo/env"
pip install .
echo "tt-smi installed!"
# run with
tt-smi
```

#### Firmware

firmware bundle: 80.10.1.0 (https://github.com/tenstorrent/tt-firmware/blob/02b4b6ed49b6ea2fb9a8664e99d4fed25e443bd6/experiments/fw_pack-80.10.1.0.fwbundle)
```bash
# check version with tt-smi
tt-smi
# press 3 to go to firmware tab, check FW Bundle Version column
# you should see the same version for all chips
```

See firmware flashing instructions at https://github.com/tenstorrent/tt-flash if needed.

#### Drivers
tt-kmd: 1.28 (https://github.com/tenstorrent/tt-kmd/tree/ttkmd-1.28)

See driver install instructions at https://github.com/tenstorrent/tt-kmd if needed.
```bash
# check for TT devices
lspci -d 1e52:
sudo lsof | grep "/dev/tenstorrent"
# clone kmd
git clone --branch ttkmd-1.28 --single-branch https://github.com/tenstorrent/tt-kmd.git 
cd tt-kmd/
# veryify not installed
modinfo tenstorrent
lsmod | grep tenstorrent
# install
sudo dkms add .
sudo dkms install tenstorrent/1.28
sudo modprobe tenstorrent
# make sure same kernel
modinfo tenstorrent
uname -a
# ONLY IF NEEDED:
# remove old version if installed
sudo modprobe -r tenstorrent
sudo dkms remove tenstorrent/1.27.1 --all
```

#### Topology
tt-topology: https://github.com/tenstorrent/tt-topology
Note: after flashing firmware, tt-topology must be run for mesh chip layout to re-establish mesh ethernet links (https://github.com/tenstorrent/tt-topology)
```bash
# check if this tt-topology command is still valid: https://github.com/tenstorrent/tt-topology?tab=readme-ov-file#mesh
tt-topology -l mesh -p mesh_layout.png
```

#### Hugepages setup

use https://github.com/tenstorrent/tt-system-tools
```bash
# Note: this is only tested on Ubuntu 20.04, may not work on different OS versions
git clone https://github.com/tenstorrent/tt-system-tools
cd tt-system-tools
sudo ./hugepages-setup.sh
```

### 3. CPU performance setting

```bash
sudo apt-get update && sudo apt-get install -y linux-tools-generic
# enable perf mode
sudo cpupower frequency-set -g performance
# disable perf mode
sudo cpupower frequency-set -g ondemand
```

### 4. Docker image build

The docker image uses tt-metal commit [ba7c8de54023579a86fde555b3c68d1a1f6c8193](https://github.com/tenstorrent/tt-metal/tree/ba7c8de54023579a86fde555b3c68d1a1f6c8193)
CI Llama 3 70B T3000 run: https://github.com/tenstorrent/tt-metal/actions/runs/10453532224/job/28944574605
```bash
## llama3 and llama2 container
docker build -t ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-ba7c8de5 . -f llama3.src.base.inference.v0.51.0-ba7c8de5.Dockerfile
docker push ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-ba7c8de5
```

### 5. download weights
To download the Llama 3.1 / 3 / 2 weights from Meta (https://llama.meta.com/llama-downloads/), you will need to submit your contact email and company information to get the license URL for downloading.

Once you have the signed URL you can run the download scripts for the respective version.

#### Download Llama 3.1 70B weights

```bash
git clone https://github.com/meta-llama/llama-models.git ~/llama3_1
export LLAMA3_1_DIR=${LLAMA3_1_DIR}/models/llama3_1
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

#### Download Llama 3 70B weights

```bash
# change this if you prefer to clone the llama3 repo elsewhere
export LLAMA3_DIR=~/llama3
git clone https://github.com/meta-llama/llama3.git $LLAMA3_DIR
cd $LLAMA3_DIR
./download.sh
# select 70B-instruct
```

Select model size `70B-instruct` and it will download to `./Meta-Llama-3-70B-Instruct`
Once the download is finished you should see the checksum message:
```log
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

Note: if other models are in persistent_volume root, and the file permissions have not been set up already you may need to change them:
```bash
cd tt-inference-server
export PERSISTENT_VOLUME_ROOT=$PWD/persistent_volume/
mkdir -p ${PERSISTENT_VOLUME_ROOT}
# this will make the files readable OUTSIDE the container via group permissions
# UID 1000 is the container user
sudo chown -R 1000:dockermount ${PERSISTENT_VOLUME_ROOT}
# set owner and group to r/w/x
sudo chmod -R 775 ${PERSISTENT_VOLUME_ROOT}
```

#### Llama 3.1 70B

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
# this will make the files readable OUTSIDE the container via group permissions
# UID 1000 is the container user
sudo chown -R 1000:dockermount ${PERSISTENT_VOLUME}
# set owner and group to r/w/x
sudo chmod -R 775 ${PERSISTENT_VOLUME}
```

#### Llama 3 70B (optional)
```bash
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama3-70bv0.0.1
# create directories in persistent volume
mkdir -p ${PERSISTENT_VOLUME}/model_weights/repacked-llama-3-70b-instruct
mkdir -p ${PERSISTENT_VOLUME}/tt_metal_cache/cache_repacked-llama-3-70b-instruct
# assuming weights are downloaded to: ~/llama3/Meta-Llama-3-70B-Instruct/
cp -r $LLAMA3_DIR/Meta-Llama-3-70B-Instruct ${PERSISTENT_VOLUME}/model_weights/llama-3-70b-instruct
# copy tokenizer and params to repacked
cp $LLAMA3_DIR/Meta-Llama-3-70B-Instruct/tokenizer.model ${PERSISTENT_VOLUME}/model_weights/repacked-llama-3-70b-instruct/tokenizer.model
cp $LLAMA3_DIR/Meta-Llama-3-70B-Instruct/params.json ${PERSISTENT_VOLUME}/model_weights/repacked-llama-3-70b-instruct/params.json
# this will make the files readable OUTSIDE the container via group permissions
# UID 1000 is the container user
sudo chown -R 1000:dockermount ${PERSISTENT_VOLUME}
# set owner and group to r/w/x
sudo chmod -R 775 ${PERSISTENT_VOLUME}
```

#### Llama 2 70B (optional)
```bash
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama2-70bv0.0.1
# create directories in persistent volume
mkdir -p ${PERSISTENT_VOLUME}/model_weights/repacked-llama-2-70b-chat
mkdir -p ${PERSISTENT_VOLUME}/tt_metal_cache/cache_repacked-llama-2-70b-chat
# assuming weights are downloaded to: ~/llama/llama-2-70b-chat
cp -r ~/llama/llama-2-70b-chat ${PERSISTENT_VOLUME}/model_weights/llama-2-70b-chat
# copy tokenizer and params to repacked
cp ~/llama/llama-2-70b-chat/tokenizer.model ${PERSISTENT_VOLUME}/model_weights/repacked-llama-2-70b-chat/tokenizer.model
cp ~/llama/llama-2-70b-chat/params.json ${PERSISTENT_VOLUME}/model_weights/repacked-llama-2-70b-chat/params.json
# this will make the files readable OUTSIDE the container via group permissions
# UID 1000 is the container user
sudo chown -R 1000:dockermount ${PERSISTENT_VOLUME}
# set owner and group to r/w/x
sudo chmod -R 775 ${PERSISTENT_VOLUME}
```

#### Repack the weights

Use the docker container to run the `repack_weights.py` script:
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
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-ba7c8de5 bash

cd /tt-metal
# need to set path environment variables for demo scripts
## llama 3.1
export LLAMA3_CKPT_DIR=/home/user/cache_root/model_weights/repacked-llama-3.1-70b-instruct
export LLAMA3_TOKENIZER_PATH=/home/user/cache_root/model_weights/repacked-llama-3.1-70b-instruct/tokenizer.model
export LLAMA3_CACHE_PATH=/home/user/cache_root/tt_metal_cache/cache_repacked-llama-3.1-70b-instruct
python models/demos/t3000/llama2_70b/scripts/repack_weights.py /home/user/cache_root/model_weights/llama-3.1-70b-instruct ${LLAMA3_CKPT_DIR} 5 80 8192
## llama 3, persistent_volume for llama3 must be mounted instead of llama3.1 volume
export LLAMA3_CKPT_DIR=/home/user/cache_root/model_weights/repacked-llama-3-70b-instruct
export LLAMA3_TOKENIZER_PATH=/home/user/cache_root/model_weights/repacked-llama-3-70b-instruct/tokenizer.model
export LLAMA3_CACHE_PATH=/home/user/cache_root/tt_metal_cache/cache_repacked-llama-3-70b-instruct
python models/demos/t3000/llama2_70b/scripts/repack_weights.py /home/user/cache_root/model_weights/llama-3-70b-instruct ${LLAMA3_CKPT_DIR} 5 80 8192
## for llama-2-70b-chat, persistent_volume for llama2 must be mounted instead of llama3 volume
export LLAMA2_CKPT_DIR=/home/user/cache_root/model_weights/repacked-llama-2-70b-instruct
export LLAMA2_TOKENIZER_PATH=/home/user/cache_root/model_weights/repacked-llama-2-70b-chat/tokenizer.model
export LLAMA2_CACHE_PATH=/home/user/cache_root/tt_metal_cache/cache_repacked-llama-2-70b-chat
python models/demos/t3000/llama2_70b/scripts/repack_weights.py /home/user/cache_root/model_weights/llama-2-70b-chat ${LLAMA2_CKPT_DIR} 5
```

# System dependencies

All system dependencies are listed and installed in `llama3.src.base.inference.v0.51.0-ba7c8de5.Dockerfile`, which references the tt-metal Dockerfiles at ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64

# Development

additionally add the src code as a volume mount so that it can be editted and rerun.

```bash
cd cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama3.1-70bv0.0.1
docker run \
  -it \
  --rm \
  --env-file tt-metal-llama3-70b/.env \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${PERSISTENT_VOLUME?ERROR env var PERSISTENT_VOLUME must be set}:/home/user/cache_root:rw \
  --volume $PWD/tt-metal-llama3-70b/src:/home/user/tt-metal-llama3-70b/src:rw \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-ba7c8de5 bash

gunicorn --config gunicorn.conf.py
```

## Run tests

### test files

`test_inference_api_alpaca_eval.py`: sends alpaca eval (https://huggingface.co/datasets/tatsu-lab/alpaca_eval) prompts to a running inference API server using HTTP requests. Tests that the inference API server + model backend are working correctly.
`test_inference_api_client_perf.py`: send prompts to a running inference API server using HTTP requests with preset number and timing. Useful for stress testing performance of server, for example when loaded beyond what tt-metal model implementation can serve.
`test_inference_api.py`: send a prompt to a running inference API server using HTTP requests, shows streaming output. Example of REST API usage.
`test_llama3_70b_backend_mock.py`: runs just the model backend without using the TT hardware or running a compute intensive model, it just sends random logits back for rapid testing. This is helpful when debugging the model backend independentof the inference API server or tt-metal model implementation.
`test_llama3_70b_backend.py`: runs just the model backend with the TT hardware model implementation. Useful for debugging model running without the inference API server.
`tt_metal_impl/eval/demo_llama3_alpaca_eval.py`: runs the llama 3 and llama 3.1 70B tt-metal demo in a loop with prompts from alpaca eval. Tests tt-metal model implementation.
`tt_metal_impl/eval/demo_llama2_alpaca_eval.py`: runs the llama 2 70B tt-metal demo in a loop with prompts from alpaca eval. Tests tt-metal model implementation.

### Test with mocks

The mock server and mock backend can be used for development on either component in isolation.
Importantly the mock implementations give a single thread synchronous implmentation for ease of debugging.

```bash
cd ~/tt-metal-llama3-70b/src
# within container, access backend mock with:
python test_llama3_70b_backend_mock.py
# access inference server mock (using backend mock) with:
python test_mock_inference_api_server.py
```

### Test with full on device backend

```bash
cd ~/tt-metal-llama3-70b/src
# test backend running on device
python test_llama3_70b_backend.py
```