# TT Metalium Whisper Distil Large v3 Inference API

This implementation supports Whisper Distil Large v3 execution on Worhmole n150 & n300.


## Table of Contents
- [Build server](#build-server)
- [Run server](#run-server)
- [JWT_TOKEN Authorization](#jwt_token-authorization)
- [Development](#development)
- [Tests](#tests)


## Build server
To build the Whisper-Distil-Large-v3 inference server, run the following command from the project root at `tt-inference-server`:
```bash
cd tt-inference-server
# source build variables
source tt-metal-whisper-distil-large-v3/.env.build
# build cloud deploy image
export CLOUD_IMAGE_TAG=ghcr.io/tenstorrent/tt-inference-server/tt-metal-whisper-distil-large-v3-cloud:${IMAGE_VERSION}-tt-metal-${TT_METAL_COMMIT_DOCKER_TAG}
docker build \
  -t ${CLOUD_IMAGE_TAG} \
  --build-arg TT_METAL_DOCKERFILE_VERSION=${TT_METAL_DOCKERFILE_VERSION} \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
  --build-arg CONTAINER_APP_UID=${CONTAINER_APP_UID} \
  . -f tt-metal-whisper-distil-large-v3/whisper-distil-large-v3.cloud.Dockerfile

# build dev image
export DEV_IMAGE_TAG=ghcr.io/tenstorrent/tt-inference-server/tt-metal-whisper-distil-large-v3-dev:${IMAGE_VERSION}-tt-metal-${TT_METAL_COMMIT_DOCKER_TAG}
docker build \
  -t ${DEV_IMAGE_TAG} \
  --build-arg CLOUD_DOCKERFILE_URL=${CLOUD_IMAGE_TAG} \
  . -f tt-metal-whisper-distil-large-v3/whisper-distil-large-v3.dev.Dockerfile
```

## Run server
To run the Whisper-Distil-Large-v3 inference server, run the following command from the project root at `tt-inference-server`:
```bash
cd tt-inference-server
# source build variables
source tt-metal-whisper-distil-large-v3/.env.build
# make sure if you already set up the model weights and cache you use the correct persistent volume
export MODEL_NAME=Whisper-Distil-Large-v3
export PERSISTENT_VOLUME_ROOT=$PWD/persistent_volume
export MODEL_VOLUME=${PERSISTENT_VOLUME_ROOT}/volume_id_tt-metal-${MODEL_NAME}-v0.0.1/
export MODEL_ENV_FILE=${PERSISTENT_VOLUME_ROOT}/model_envs/${MODEL_NAME}.env
docker run \
  --rm \
  -it \
  --env-file ${MODEL_ENV_FILE} \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${MODEL_VOLUME?ERROR env var MODEL_VOLUME must be set}:/home/container_app_user/cache_root:rw \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-whisper-distil-large-v3-dev:${IMAGE_VERSION}-tt-metal-${TT_METAL_COMMIT_DOCKER_TAG}
```

This will start the default Docker container with the entrypoint command set to run the gunicorn server. The next section describes how to override the container's default command with an interractive shell via `bash`.


### JWT_TOKEN Authorization

To authenticate requests use the header `Authorization`. The JWT token can be computed using the script `jwt_util.py`. This is an example:
```bash
cd tt-inference-server/tt-metal-yolov4/server
export JWT_SECRET=<your-secure-secret>
export AUTHORIZATION="Bearer $(python scripts/jwt_util.py --secret ${JWT_SECRET?ERROR env var JWT_SECRET must be set} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}')"
```


## Development
To interactively develop within the container, [ensure it has been built](#build-server) then start the container with:
Inside the container you can then start the server with:
```bash
cd tt-inference-server
# source build variables
source tt-metal-whisper-distil-large-v3/.env.build
# make sure if you already set up the model weights and cache you use the correct persistent volume
export MODEL_NAME=Whisper-Distil-Large-v3
export PERSISTENT_VOLUME_ROOT=$PWD/persistent_volume
export MODEL_VOLUME=${PERSISTENT_VOLUME_ROOT}/volume_id_tt-metal-${MODEL_NAME}-v0.0.1/
export MODEL_ENV_FILE=${PERSISTENT_VOLUME_ROOT}/model_envs/${MODEL_NAME}.env
docker run \
  --rm \
  -it \
  --env-file ${MODEL_ENV_FILE} \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${MODEL_VOLUME?ERROR env var MODEL_VOLUME must be set}:/home/container_app_user/cache_root:rw \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-whisper-distil-large-v3-dev:${IMAGE_VERSION}-tt-metal-${TT_METAL_COMMIT_DOCKER_TAG}
  /bin/bash
```

Inside the container, run `cd ~/app/server` to navigate to the server implementation.


## Tests
Tests can be found in `tests/`. The tests have their own dependencies found in `requirements-test.txt`.

First, ensure the server is running (see [how to run the server](#run-server)). Then in a different shell with the base dev `venv` activated:
```bash
cd tt-metal-whisper-distil-large-v3
pip install -r requirements-test.txt
cd tests/
pytest test_inference_api.py -s
```
