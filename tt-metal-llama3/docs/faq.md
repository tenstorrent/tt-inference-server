# Frequently Asked Questions

## How to do a soft reset?

On host, use tt-smi (https://github.com/tenstorrent/tt-smi) to reset the n300 devices: 
```bash
# source
source ~/.venv/bin/activate
# note the device ids may be different
# check: ls /dev/tenstorrent/
tt-smi -r 0,1,2,3
```

This soft reset is required for example when the device is not closed correctly during termination.
When this occurs the device may not be able to connect and train the ethernet links. If this occurs try soft resetting the device.

## How to change inference server port?
if you want to change the port, change SERVICE_PORT (defaut 7000) and the Docker published port mapping (default is `--publish 7000:7000`)

## How to get JWT Token without the script?

For example, without using the `src/scripts/jwt_util.py` script:

```bash
pip install pyjwt==2.7.0
```

Then run the python code in an interactive python shell:
```python
import json
import jwt
jwt_secret = <your-secure-secret>
json_payload = json.loads('{"team_id": "tenstorrent", "token_id":"debug-test"}')
encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
print(encoded_jwt)
```

## From host send requests using alpaca eval

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

## What are the system dependencies?

All system dependencies are listed and installed in `llama3.src.base.inference.v0.51.0-ba7c8de5.Dockerfile`, which itself is based on the tt-metal Dockerfiles at ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64.

## Llama 3 setup differences

download Llama 3 70B weights
```bash
# change this if you prefer to clone the llama3 repo elsewhere
export LLAMA3_DIR=~/llama3
git clone https://github.com/meta-llama/llama3.git ${LLAMA3_DIR}
cd ${LLAMA3_DIR}
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

download weights:
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
repack:
```bash
## llama 3, persistent_volume for llama3 must be mounted instead of llama3.1 volume
export LLAMA3_CKPT_DIR=/home/user/cache_root/model_weights/repacked-llama-3-70b-instruct
export LLAMA3_TOKENIZER_PATH=/home/user/cache_root/model_weights/repacked-llama-3-70b-instruct/tokenizer.model
export LLAMA3_CACHE_PATH=/home/user/cache_root/tt_metal_cache/cache_repacked-llama-3-70b-instruct
python models/demos/t3000/llama2_70b/scripts/repack_weights.py /home/user/cache_root/model_weights/llama-3-70b-instruct ${LLAMA3_CKPT_DIR} 5
```

```bash
# need to set path environment variables for demo scripts using different weights
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama3-70bv0.0.1
export LLAMA_VERSION=llama3
export LLAMA3_CKPT_DIR=/home/user/cache_root/model_weights/repacked-llama-3-70b-instruct
export LLAMA3_TOKENIZER_PATH=/home/user/cache_root/model_weights/repacked-llama-3-70b-instruct/tokenizer.model
export LLAMA3_CACHE_PATH=/home/user/cache_root/tt_metal_cache/cache_repacked-llama-3-70b-instruct
```

## Llama 2 setup differences

download weights:
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

repack:
```bash
## for llama-2-70b-chat, persistent_volume for llama2 must be mounted instead of llama3 volume
export LLAMA2_CKPT_DIR=/home/user/cache_root/model_weights/repacked-llama-2-70b-instruct
export LLAMA2_TOKENIZER_PATH=/home/user/cache_root/model_weights/repacked-llama-2-70b-chat/tokenizer.model
export LLAMA2_CACHE_PATH=/home/user/cache_root/tt_metal_cache/cache_repacked-llama-2-70b-chat
python models/demos/t3000/llama2_70b/scripts/repack_weights.py /home/user/cache_root/model_weights/llama-2-70b-chat ${LLAMA2_CKPT_DIR} 5
```

```bash
# need to set path environment variables for demo scripts using different weights
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama2-70bv0.0.1
export LLAMA_VERSION=llama2
export LLAMA3_CKPT_DIR=/home/user/cache_root/model_weights/repacked-llama-2-70b-instruct
export LLAMA3_TOKENIZER_PATH=/home/user/cache_root/model_weights/repacked-llama-2-70b-instruct/tokenizer.model
export LLAMA3_CACHE_PATH=/home/user/cache_root/tt_metal_cache/cache_repacked-llama-2-70b-instruct
```

