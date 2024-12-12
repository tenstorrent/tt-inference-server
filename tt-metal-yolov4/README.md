# TT Metalium YoloV4 Inference API

This implementation supports YoloV4 execution on Grayskull and Worhmole.


## Table of Contents
- [Run server](#run-server)
- [Development](#development)
- [Tests](#tests)


## Run server
To run the YoloV4 inference server, run the following command from the project root at `tt-inference-server`:
```bash
cd tt-inference-server
docker compose --env-file tt-metal-yolov4/.env.default -f tt-metal-yolov4/docker-compose.yaml up --build
```

This will start the default Docker container with the entrypoint command set to `server/run_uvicorn.sh`. The next section describes how to override the container's default command with an interractive shell via `bash`.


## Development
Inside the container you can then start the server with:
```bash
docker compose --env-file tt-metal-yolov4/.env.default -f tt-metal-yolov4/docker-compose.yaml run --rm inference_server /bin/bash
```

Inside the container, run `cd ~/app/server` to navigate to the server implementation.


## Tests
Tests can be found in `tests/`. The tests have their own dependencies found in `requirements-test.txt`.

To load test the server, we use `locust` to simulate a single client sending an infinite-FPS video stream to the server for 1 minute.
This yields a server performance ceiling of ~25FPS. First, ensure the server is running (see [how to run the server](#run-server)). Then in a different shell with the base dev `venv` activated:
```bash
cd tt-metal-yolov4
pip install -r requirements-test.txt
cd tests/
locust --config locust_config.conf
```
