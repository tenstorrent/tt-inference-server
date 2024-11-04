# Test vLLM via Mock Model 

This directory contains scripts to allow for rapid testing and development of benchmarking, evaluation, and stress testing procedures for available models through [vllm](https://github.com/tenstorrent/vllm/tree/dev) 

To run the mock offline inference script `mock_vllm_offline_inference_tt.py` follow the steps below:

## 1. Build Docker Container

Follow instructions under `evals/README.md` to build the docker container. To set the environment variables appropriately for the latest supported versions of tt-metal and vllm, refer to [vllm/tt_metal](https://github.com/tenstorrent/vllm/blob/dev/tt_metal/README.md) when setting: 

```bash
export TT_METAL_COMMIT_SHA_OR_TAG=<tt-mettal-commit>
export VLLM_COMMIT_SHA=<vllm-commit>
```

## 2. Run The Docker Container

Add a volume mounting the `test` directory in the container before running with the following in the docker run command:

```bash
--volume $PWD/tests:/home/user/tests
```

## 3. Run The Mock Model

Once in the docker container, run the mock script with:

```bash
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python /home/user/tests/mock_vllm_offline_inference_tt.py
```