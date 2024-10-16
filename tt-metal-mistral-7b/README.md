# TT Metalium Mistral 7B Inference API

This implementation support the model: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

## Quick Run

If you're starting from scratch or the quick run is not working see the [setup](#setup) section below. Otherwise for a quick run follow the below:


### Docker Run - Mistral 7B inference server
```bash
# create JWT secret to later be used to authenticate user requests via the decoding of Authorization
export JWT_SECRET=<your-secure-secret>
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-mistral-7bv0.0.2
docker run \
  --rm \
  -it \
  --env-file tt-metal-mistral-7b/.env.default \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --env JWT_SECRET=${JWT_SECRET} \
  --env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml \
  --env SERVICE_PORT=7000 \
  --volume ${PERSISTENT_VOLUME}:/home/user/cache_root:rw \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ./tt-metal-mistral-7b/src:/mnt/src \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-mistral-7b-src-base:v0.0.3-tt-metal-v0.52.0-rc33 

```
in another terminal
```bash
export AUTHORIZATION="Bearer $(python scripts/jwt_util.py --secret ${JWT_SECRET?ERROR env var JWT_SECRET must be set} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}')"
python test_inference_api.py
```

## Setup

Tested starting condition is from a fresh installation of Ubuntu 20.04 with Tenstorrent system dependencies installed. 


#### build docker container (for development)

```bash
cd tt-inference-server
cd tt-metal-mistral-7b
docker build -t ghcr.io/tenstorrent/tt-inference-server/tt-metal-mistral-7b-src-base:v0.0.1-tt-metal-v0.52.0-rc33 -f mistral7b.src.base.inference.v0.52.0-rc33.Dockerfile .

# build with code server
docker build -t ghcr.io/tenstorrent/tt-inference-server/tt-metal-mistral-7b-src-base:v0.0.1-tt-metal-v0.51.0-rc29-cs -f mistral7b.src.base.inference.v0.51.0-rc29-cs.Dockerfile .
```

Or to use a pre-built image pull the following image from the GitHub Container Registry (GHCR):
```bash
docker pull ghcr.io/tenstorrent/tt-inference-server/tt-metal-mistral-7b-src-base:v0.0.3-tt-metal-v0.52.0-rc33
```
#### Setup file permissions for docker usage (for development)
```bash
cd tt-inference-server
mkdir persistent_volume/volume_id_tt-metal-mistral-7bv0.0.2
# use a group e.g. named "dockermount" that has access to the files outside the container
# and your host user is a member of
sudo groupadd dockermount
sudo usermod -aG dockermount <host_user>
# you will likely need to reset group data in current shell
# this will be needed in new shells until next logout/login from your user (you can do that now alternatively)
sudo newgrp dockermount
sudo chown -R 1000:dockermount persistent_volume/volume_id_tt-metal-mistral-7bv0.0.2
# make sure the group write bit is set for write permissions
sudo chmod -R g+w persistent_volume/volume_id_tt-metal-mistral-7bv0.0.2
```

### Docker run (for development)

```bash
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export JWT_SECRET=<your-secure-secret>
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-mistral-7bv0.0.2
docker run \
  --rm \
  -it \
  --env-file tt-metal-mistral-7b/.env.default \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --env JWT_SECRET=${JWT_SECRET} \
  --env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml \
  --env SERVICE_PORT=7000 \
  --volume ${PERSISTENT_VOLUME}:/home/user/cache_root:rw \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ./tt-metal-mistral-7b/src:/mnt/src \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-mistral-7b-src-base:v0.0.3-tt-metal-v0.52.0-rc33 
```

#### Download weights

Follow instructions to download weights and setup for either general `Mistral-7B-v0.1` or instruct fine-tuned `Mistral-7B-Instruct-v0.2` from https://github.com/tenstorrent/tt-metal/blob/main/models/demos/wormhole/mistral7b/README.md

```bash
# inside container
export MISTRAL_CACHE_PATH=/home/user/cache_root/tt_metal_cache/mistral-7B-instruct-v0.2
export MISTRAL_CKPT_DIR=/home/user/cache_root/model_weights/mistral-7B-instruct-v0.2
export MISTRAL_TOKENIZER_PATH=/home/user/cache_root/model_weights/mistral-7B-instruct-v0.2
mkdir -p ${MISTRAL_CKPT_DIR}
mkdir -p ${MISTRAL_CACHE_PATH}
python /tt-metal/models/demos/wormhole/mistral7b/scripts/get_mistral_weights.py --weights_path=${MISTRAL_CKPT_DIR} --instruct
```

#### Start Gunicorn Server
You can now start the gunicorn server
```bash
gunicorn --config gunicorn.conf.py
```

#### JWT_TOKEN Authorization + Test the inference server

To authenticate requests use the header Authorization. The JWT token can be computed using the script jwt_util.py. This is an example:

To send HTTP requests to the inference server run the example scripts in a separate bash shell. You can use `docker exec -it <container-id> bash` to create a shell in the docker container or run the client scripts on the host ensuring the correct port mappings and python dependencies are available:
```bash
export AUTHORIZATION="Bearer $(python scripts/jwt_util.py --secret ${JWT_SECRET?ERROR env var JWT_SECRET must be set} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}')"
python test_inference_api.py
```
