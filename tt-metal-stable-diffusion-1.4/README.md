# TT Metalium Stable Diffusion 1.4 Inference API

This implementation supports Stable Diffusion 1.4 execution on Worhmole n150 (n300 currently broken).


## Table of Contents
- [Run server](#run-server)
- [JWT_TOKEN Authorization](#jwt_token-authorization)
- [Development](#development)
- [Tests](#tests)


## Run server
To run the SD1.4 inference server, run the following command from the project root at `tt-inference-server`:
```bash
cd tt-inference-server
docker compose --env-file tt-metal-stable-diffusion-1.4/.env.default -f tt-metal-stable-diffusion-1.4/docker-compose.yaml up --build
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
docker compose --env-file tt-metal-stable-diffusion-1.4/.env.default -f tt-metal-stable-diffusion-1.4/docker-compose.yaml run --rm --build inference_server /bin/bash
```

Inside the container, run `cd ~/app/server` to navigate to the server implementation.


## Tests
Tests can be found in `tests/`. The tests have their own dependencies found in `requirements-test.txt`.

First, ensure the server is running (see [how to run the server](#run-server)). Then in a different shell with the base dev `venv` activated:
```bash
cd tt-metal-stable-diffusion-1.4
pip install -r requirements-test.txt
cd tests/
locust --config locust_config.conf
```
