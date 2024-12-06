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

To load test the server, we use `locust` to simulate multiple clients sending 15FPS video streams to the server. The test can be run the following command:
```bash

```
