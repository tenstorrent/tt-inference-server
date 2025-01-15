# Development vllm-tt-metal-llama3

Containerization in: https://github.com/tenstorrent/tt-inference-server/blob/tstesco/vllm-llama3-70b/vllm-tt-metal-llama3/vllm.llama3.src.base.inference.v0.52.0.Dockerfile 

tt-metal and vLLM are under active development in lock-step: https://github.com/tenstorrent/vllm/tree/dev/tt_metal 

lm-evaluation-harness fork: https://github.com/tstescoTT/lm-evaluation-harness

## Step 1: Build Docker Image

When building, update the commit SHA and get correct SHA from model developers or from vLLM readme (https://github.com/tenstorrent/vllm/tree/dev/tt_metal#vllm-and-tt-metal-branches ). The Dockerfile version updates infrequently but may also be updated.
```bash
# set build context to repo root
cd tt-inference-server
# build image
export TT_METAL_DOCKERFILE_URL=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64:v0.53.0-rc34-dev
export TT_METAL_COMMIT_SHA_OR_TAG=v0.54.0-rc2
export TT_METAL_COMMIT_DOCKER_TAG=${TT_METAL_COMMIT_SHA_OR_TAG:0:12}
export TT_VLLM_COMMIT_SHA_OR_TAG=953161188c50f10da95a88ab305e23977ebd3750
export TT_VLLM_COMMIT_DOCKER_TAG=${TT_VLLM_COMMIT_SHA_OR_TAG:0:12}
export IMAGE_VERSION=v0.0.1
docker build \
  -t ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm-${OS_VERSION}:${IMAGE_VERSION}-${TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG} \
  --build-arg TT_METAL_DOCKERFILE_URL=${TT_METAL_DOCKERFILE_URL} \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
  --build-arg TT_VLLM_COMMIT_SHA_OR_TAG=${TT_VLLM_COMMIT_SHA_OR_TAG} \
  . -f vllm-tt-metal-llama3/vllm.llama3.src.Dockerfile
```

### Ubuntu 22.04 base image

In the tt-metal repo there is a Ubuntu 22.04 Dockerfile: https://github.com/tenstorrent/tt-metal/blob/main/dockerfile/ubuntu-22.04-amd64.Dockerfile
This Dockerfile installs the python dependencies for Ubuntu 22.04 running Python 3.10: https://github.com/tenstorrent/tt-metal/blob/main/scripts/docker/requirements-22.04.txt

The Ubuntu 22.04 images are not yet published to GHCR as the Ubuntu 20.04 images are (https://github.com/tenstorrent/tt-metal/pkgs/container/tt-metal%2Ftt-metalium%2Fubuntu-20.04-amd64)

You can build local tt-metal ubuntu 22.04 base image:
```bash
git clone --depth 1 --branch ${TT_METAL_COMMIT_SHA_OR_TAG} https://github.com/tenstorrent/tt-metal.git
cd tt-metal
docker build -t local/tt-metal/tt-metalium/ubuntu-22.04-amd64:latest -f dockerfile/ubuntu-22.04-amd64.Dockerfile .
```

You can then repeat the steps above to build with, e.g. `TT_METAL_DOCKERFILE_URL=local/tt-metal/tt-metalium/ubuntu-22.04-amd64:latest`

### push image (only for admin deployment to GHCR)
```bash
docker push ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm-${OS_VERSION}:${IMAGE_VERSION}-${TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG}
```

## Step 2: Run container for LM evals development

note: this requires running `setup.sh` to set up the weights for a particular model, in this example `llama-3.1-70b-instruct`.

```bash
cd tt-inference-server
export MODEL_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-llama-3.1-70b-instructv0.0.1/
docker run \
  --rm \
  -it \
  --env-file persistent_volume/model_envs/llama-3.1-70b-instruct.env \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${MODEL_VOLUME?ERROR env var MODEL_VOLUME must be set}:/home/user/cache_root:rw \
  --shm-size 32G \
  ghcr.io/tenstorrent/tt-inference-server/tt-metal-llama3-70b-src-base-vllm:v0.0.1-tt-metal-${TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG} bash
```

additionally for development you can mount the volumes:
```bash
  --volume $PWD/../vllm:/home/user/vllm \
  --volume $PWD/../lm-evaluation-harness:/home/user/lm-evaluation-harness \
```

## Step 3: Inside container setup and run vLLM

The following env vars are required be set by the Dockerfile already:

- `PYTHON_ENV_DIR="${TT_METAL_HOME}/python_env"`: location where tt-metal python environment was installed. This is defined in Dockerfile.
- `VLLM_TARGET_DEVICE="tt"`: This is defined in Dockerfile.
- `vllm_dir`: Location of vLLM installation. This is defined in Dockerfile. You must update this if you've changed the vLLM install locationed.

#### Option 1: use default installation in docker image

Already built into Docker image, continue to run vLLM.

#### option 2: install vLLM from github

```bash
# option 2: install from github
cd /home/user/vllm
git fetch
git checkout <branch>
git pull
pip install -e .
echo "done vllm install."
```
#### option 3: install edittable (for development) - mount from outside container

```bash
# option 3: install edittable (for development) - mount from outside container
cd /home/user/vllm
pip install -e .
echo "done vllm install."
```

#### Run vllm serving openai compatible API server

```bash
# run vllm serving
cd /home/user/vllm
python examples/server_example_tt.py
```

## Sending requests to vLLM inference server

If the container is exposing a port (e.g. `docker run ... --publish 7000:7000`), you can send requests to that port , otherwise you can enter an interactive shell within the container via:
```bash
docker exec -it $(docker ps -q | head -n1) bash
```
