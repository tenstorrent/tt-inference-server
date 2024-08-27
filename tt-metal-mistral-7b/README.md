# TT Metalium Mistral 7B Inference API

Model: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

## Quick Run

If you're starting from scratch or the quick run is not working see the [setup][#setup] section below.

### tt-metal model demo
```bash
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
pytest models/demos/wormhole/mistral7b/demo/demo_with_prefill.py::test_mistral7B_demo[general_weights-<number_of_batches>_batch]
```

### inference server

start the gunicorn server:
```bash
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
gunicorn --config gunicorn.conf.py
```

in another shell:
```bash
python src/test_inference_api.py
```

## setup

If you're running inside a container already, you can skip the Docker image set up steps (for development).

#### build docker container (for development)

```bash
cd tt-inference-server
cd tt-metal-mistral-7b
docker build -t ghcr.io/tenstorrent/tt-inference-server/tt-metal-mistral-7b-src-base:v0.0.1-tt-metal-v0.51.0-rc29 -f mistral7b.src.base.inference.v0.51.0-rc29.Dockerfile .

# build with code server
docker build -t ghcr.io/tenstorrent/tt-inference-server/tt-metal-mistral-7b-src-base:v0.0.1-tt-metal-v0.51.0-rc29-cs -f mistral7b.src.base.inference.v0.51.0-rc29-cs.Dockerfile .
```

#### setup file permissions for docker usage (for development)
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

### docker run (for development)

```bash
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-mistral-7bv0.0.1
export JWT_SECRET=<your-secret>
docker run \
  --rm \
  -it \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --env JWT_SECRET=${JWT_SECRET} \
  --env CACHE_ROOT=/mnt/cache_root \
  --env HF_HOME=/mnt/cache_root/huggingface \
  --env TT_METAL_ASYNC_DEVICE_QUEUE=1 \
  --env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml \
  --env SERVICE_PORT=7000 \
  --env MISTRAL_CKPT_DIR=/mnt/cache_root/model_weights/mistral-7B-v0.1 \
  --env MISTRAL_TOKENIZER_PATH=/mnt/cache_root/model_weights/mistral-7B-v0.1 \
  --env MISTRAL_CACHE_PATH=/mnt/cache_root/tt_metal_cache/mistral-7B-v0.1 \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${PERSISTENT_VOLUME}:/mnt/cache_root:rw \
  --volume ./tt-metal-mistral-7b/src:/mnt/src \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-mistral-7b-src-base:v0.0.1-tt-metal-v0.51.0-rc29 bash
```

#### download weights

Follow instructions to download weights and setup for either general `Mistral-7B-v0.1` or instruct fine-tuned `Mistral-7B-Instruct-v0.2` from https://github.com/tenstorrent/tt-metal/blob/main/models/demos/wormhole/mistral7b/README.md

```bash
# inside container
export MISTRAL_CKPT_DIR=/mnt/cache_root/model_weights/mistral-7B-v0.1
python /tt-metal/models/demos/wormhole/mistral7b/scripts/get_mistral_weights.py --weights_path=$MISTRAL_CKPT_DIR --instruct
```

