# TT Metalium Stable Diffusion 3.5 Inference API

This implementation supports Stable Diffusion 3.5 execution on Worhmole n150 & n300.


## Table of Contents
- [Run server](#run-server)
- [JWT_TOKEN Authorization](#jwt_token-authorization)
- [Development](#development)
- [Tests](#tests)


## Run server
To run the SD3.5 inference server, run the following command from the project root at `tt-inference-server`:
```bash
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export MODEL_NAME=Stable-Diffusion-3.5-medium
export PERSISTENT_VOLUME_ROOT=$PWD/persistent_volume
export MODEL_VOLUME=${PERSISTENT_VOLUME_ROOT}/volume_id_tt-metal-${MODEL_NAME}-v0.0.1/
export MODEL_ENV_FILE=${PERSISTENT_VOLUME_ROOT}/model_envs/${MODEL_NAME}.env
docker compose --env-file tt-metal-stable-diffusion-3.5/.env.default -f tt-metal-stable-diffusion-3.5/docker-compose.yaml up --build
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
Inside the container you can then start the server with:
```bash
cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export MODEL_NAME=Stable-Diffusion-3.5-medium
export PERSISTENT_VOLUME_ROOT=$PWD/persistent_volume
export MODEL_VOLUME=${PERSISTENT_VOLUME_ROOT}/volume_id_tt-metal-${MODEL_NAME}-v0.0.1/
export MODEL_ENV_FILE=${PERSISTENT_VOLUME_ROOT}/model_envs/${MODEL_NAME}.env
docker compose --env-file tt-metal-stable-diffusion-3.5/.env.default -f tt-metal-stable-diffusion-3.5/docker-compose.yaml run --rm --build inference_server /bin/bash
```

Inside the container, run `cd ~/app/server` to navigate to the server implementation.


## Tests
Tests can be found in `tests/`. The tests have their own dependencies found in `requirements-test.txt`.

First, ensure the server is running (see [how to run the server](#run-server)). Then in a different shell with the base dev `venv` activated:
```bash
cd tt-metal-stable-diffusion-3.5
pip install -r requirements-test.txt
cd tests/
locust --config locust_config.conf
```
