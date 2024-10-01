# Manual Setup Guide
This manual setup is not recommended if you can use the `setup.sh` script. This method is alternative to setup.sh, and can be configured as needed. You can look at the code in `setup.sh` to get an idea on how this works as well.

__WARNING:__ Testing of alternative paths, permissions, and weights formats is not done and may require debugging.

## 1. Set up environment variables

```bash
cd tt-inference-server/tt-metal-llama3-70b
cp .env.default .env
# set JWT_SECRET to a strong secret for production environments
vim .env
# .env is ignored from git
```

## 2. download weights
To download the Llama 3.1 weights from Meta (https://llama.meta.com/llama-downloads/), you will need to submit your contact email and company information to get the license URL for downloading.

Once you have the signed URL you can run the download scripts for the respective version.

## 3. Download Llama 3.1 70B weights

```bash
export LLAMA3_1_REPO=~/llama-models
git clone https://github.com/meta-llama/llama-models.git ${LLAMA3_1_REPO}
export LLAMA3_1_DIR=${LLAMA3_1_REPO}/models/llama3_1
cd ${LLAMA3_1_DIR}
# checkout commit before ./download.sh was removed
git checkout 685ac4c107c75ce8c291248710bf990a876e1623
./download.sh
# input meta-llama-3.1-70b
# then, input meta-llama-3.1-70b-instruct
```

NOTE: The input to ./download.sh for instruct fine-tuned models is different, you must first input `meta-llama-3.1-70b`, then input `meta-llama-3.1-70b-instruct`:
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

## 4. set up directories and permissions

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

## 5. copy over weights, tokenizer, params

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

## 6. Repack the weights

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
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.52.0-rc31-9d3be887987b bash

cd /tt-metal
python models/demos/t3000/llama2_70b/scripts/repack_weights.py /home/user/cache_root/model_weights/llama-3.1-70b-instruct ${LLAMA3_CKPT_DIR} 5
```
