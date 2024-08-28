# Development

additionally add the src code as a volume mount so that it can be editted and rerun.

```bash
cd cd tt-inference-server
# make sure if you already set up the model weights and cache you use the correct persistent volume
export PERSISTENT_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama3.1-70bv0.0.1
docker run \
  -it \
  --rm \
  --env-file tt-metal-llama3-70b/.env \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${PERSISTENT_VOLUME?ERROR env var PERSISTENT_VOLUME must be set}:/home/user/cache_root:rw \
  --volume $PWD/tt-metal-llama3-70b/src:/home/user/tt-metal-llama3-70b/src:rw \
  --shm-size 32G \
  --publish 7000:7000 \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-ba7c8de5 bash

gunicorn --config gunicorn.conf.py
```

## Run tests

### test files

`test_inference_api_alpaca_eval.py`: sends alpaca eval (https://huggingface.co/datasets/tatsu-lab/alpaca_eval) prompts to a running inference API server using HTTP requests. Tests that the inference API server + model backend are working correctly.
`test_inference_api_client_perf.py`: send prompts to a running inference API server using HTTP requests with preset number and timing. Useful for stress testing performance of server, for example when loaded beyond what tt-metal model implementation can serve.
`test_inference_api.py`: send a prompt to a running inference API server using HTTP requests, shows streaming output. Example of REST API usage.
`test_llama3_70b_backend_mock.py`: runs just the model backend without using the TT hardware or running a compute intensive model, it just sends random logits back for rapid testing. This is helpful when debugging the model backend independentof the inference API server or tt-metal model implementation.
`test_llama3_70b_backend.py`: runs just the model backend with the TT hardware model implementation. Useful for debugging model running without the inference API server.
`test_mock_inference_api_server.py`: run inference API server without using the TT hardware or running a compute intensive model. Useful for testing inference server implementation.
`scripts/demo_llama3_alpaca_eval.py`: runs the llama 3 and llama 3.1 70B tt-metal demo in a loop with prompts from alpaca eval. Tests tt-metal model implementation.
`scripts/demo_llama2_alpaca_eval.py`: runs the llama 2 70B tt-metal demo in a loop with prompts from alpaca eval. Tests tt-metal model implementation.

### Test with mocks

The mock server and mock backend can be used for development on either component in isolation.
Importantly the mock implementations give a single thread synchronous implmentation for ease of debugging.

```bash
cd ~/tt-metal-llama3-70b/src
# within container, access backend mock with:
python test_llama3_70b_backend_mock.py
# test inference server mock (using backend mock) with:
python test_mock_inference_api_server.py
```

### Test with full on device backend

```bash
cd ~/tt-metal-llama3-70b/src
# test backend running on device
python test_llama3_70b_backend.py
```

## Docker build and push

The docker image uses tt-metal commit [ba7c8de54023579a86fde555b3c68d1a1f6c8193](https://github.com/tenstorrent/tt-metal/tree/ba7c8de54023579a86fde555b3c68d1a1f6c8193)
CI Llama 3 70B T3000 run: https://github.com/tenstorrent/tt-metal/actions/runs/10453532224/job/28944574605
```bash
## build and push image
docker build -t ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-ba7c8de5 . -f llama3.src.base.inference.v0.51.0-ba7c8de5.Dockerfile
docker push ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-inference:v0.0.1-tt-metal-v0.51.0-ba7c8de5
```