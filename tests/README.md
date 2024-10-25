# Test VLLM via Mock Model 

This directory contains scripts to allow for rapid testing and development of benchmarking, evaluation,and stress testing procedures for available models through [vllm](https://github.com/tenstorrent/vllm/tree/dev) 

To run the mock offline inference script `mock_vllm_offline_inference_tt.py` follow the steps below:

## 1. Build Docker Container

Follow instructions under `evals/README.md` to build the docker container. Currently the following commit at main of [tt-metal](https://github.com/tenstorrent/tt-metal/tree/01070409e582616d8962f371e8497abbf252bb81) is supported: `01070409e582616d8962f371e8497abbf252bb81` . The corresponding [vllm](https://github.com/tenstorrent/vllm/commit/e6e80f7e63fd434e367584e726995d820023ae14) commit is on dev at: `e6e80f7e63fd434e367584e726995d820023ae14`.


Set the following enviroment variables to the appropriate commits as development goes on. 

```bash
export TT_METAL_COMMIT_SHA_OR_TAG=01070409e582616d8962f371e8497abbf252bb81
export VLLM_COMMIT_SHA=e6e80f7e63fd434e367584e726995d820023ae14

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