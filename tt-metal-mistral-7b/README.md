# TT Metalium Mistral 7B Inference API

Model: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2


### setup

#### setup weights and file permissions
```bash
cd tt-inference-server
mkdir persistent_volume/volume_id_tt-metal-mistral-7bv0.0.1
# use a group e.g. named "dockermount" that has access to the files outside the container
# and your host user is a member of
sudo groupadd dockermount
sudo usermod -aG dockermount <host_user>
# you will likely need to reset group data in current shell
# this will be needed in new shells until next logout/login from your user (you can do that now alternatively)
sudo newgrp dockermount
sudo chown -R 1000:dockermount persistent_volume/volume_id_tt-metal-mistral-7bv0.0.1
# make sure the group write bit is set for write permissions
sudo chmod -R g+w persistent_volume/volume_id_tt-metal-mistral-7bv0.0.1
```

#### build docker container

```bash
cd tt-inference-server/tt-metal-mistral-7b
docker build -t ghcr.io/tenstorrent/tt-inference-server/tt-metal-mistral-7b-src-base:v0.0.1-tt-metal-v0.51.0-rc29 -f mistral7b.src.base.inference.v0.51.0-rc29.Dockerfile .
```

#### push docker container

```bash

```

### run

```bash
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-mistral-7bv0.0.1
docker run \
  --rm \
  -it \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --env JWT_SECRET=test-secret-456 \
  --env CACHE_ROOT=/mnt/cache_root \
  --env HF_HOME=/mnt/huggingface \
  --env TT_METAL_ASYNC_DEVICE_QUEUE=1 \
  --env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml \
  --env SERVICE_PORT=7000 \
  --env PYTHONPATH="/tt-metal" \
  --env PASSWORD="password" \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${PERSISTENT_VOLUME}:/mnt/cache_root:rw \
  --env MISTRAL_CKPT_DIR=/mnt/cache_root/mistral-7B-v0.1 \
  --env MISTRAL_TOKENIZER_PATH=/mnt/cache_root/mistral-7B-v0.1 \
  --env MISTRAL_CACHE_PATH=/mnt/cache_root/mistral-7B-v0.1 \
  --volume ./tt-metal-mistral-7b/src:/mnt/src \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-mistral-7b-src-base:v0.0.1-tt-metal-v0.51.0-rc29 bash
```


  --env MODEL_WEIGHTS_ID=id_ \
  --env MODEL_WEIGHTS_PATH=/mnt/model_weights/ \