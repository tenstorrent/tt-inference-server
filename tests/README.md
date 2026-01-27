# Development Tests

This directory contains test suites and scripts for testing and development of the tt-inference-server.

## Running Tests with pytest

To run the development tests, you'll need to install the development dependencies and then use pytest.

### 1. Install Development Requirements

Install the development dependencies from the root of the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -r tt-media-server/requirements.txt
```

This will install:
- `pytest` for running tests
- `ruff` for code linting and formatting
- `pre-commit` for git hooks

### 2. Run Tests

From the project root directory, run all tests:

```bash
pytest
```

Run tests with verbose output:

```bash
pytest tests/ -v
```

Run specific test files:

```bash
pytest tests/test_run_arguments.py -v
```

Run tests with live logging output:

```bash
pytest tests/ -v --log-cli-level=INFO
```

### 3. Test Structure

The test suite includes:
- `test_run_arguments.py`: Tests for command-line argument parsing and validation in `run.py`
- `test_vllm_seq_lens.py`: [WIP] Tests for vLLM sequence length benchmarking functionality

### 4. Development Workflow

For development, you can also set up pre-commit hooks to automatically run linting and formatting:

```bash
pre-commit install
```

This will run `ruff` checks before each commit to maintain code quality.



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
--volume $PWD/tests:/home/container_app_user/tests
```

## 3. Run The Mock Model

Once in the docker container, run the mock script with:

```bash
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python /home/container_app_user/tests/mock_vllm_offline_inference_tt.py
```

# Build mock model container

```bash
# set build context to repo root
cd tt-inference-server
# build image
export TT_METAL_DOCKERFILE_URL=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64:v0.53.0-rc34-dev
export TT_METAL_COMMIT_SHA_OR_TAG=385904186f81fed15d5c87c162221d4f34387164
export TT_METAL_COMMIT_DOCKER_TAG=${TT_METAL_COMMIT_SHA_OR_TAG:0:12}
export TT_VLLM_COMMIT_SHA_OR_TAG=384f1790c3be16e1d1b10de07252be2e66d00935
export TT_VLLM_COMMIT_DOCKER_TAG=${TT_VLLM_COMMIT_SHA_OR_TAG:0:12}
docker build \
  -t ghcr.io/tenstorrent/tt-inference-server/mock.vllm.openai.api:v0.0.1-tt-metal-${TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG} \
  --build-arg TT_METAL_DOCKERFILE_URL=${TT_METAL_DOCKERFILE_URL} \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
  --build-arg TT_VLLM_COMMIT_SHA_OR_TAG=${TT_VLLM_COMMIT_SHA_OR_TAG} \
  . -f tests/mock.vllm.openai.api.dockerfile
```
