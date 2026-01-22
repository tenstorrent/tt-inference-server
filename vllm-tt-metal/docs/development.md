# Development vllm-tt-metal

Containerization in: https://github.com/tenstorrent/tt-inference-server/blob/tstesco/vllm-llama3-70b/vllm-tt-metal/vllm.llama3.src.base.inference.v0.52.0.Dockerfile 

tt-metal and vLLM are under active development in lock-step: https://github.com/tenstorrent/vllm/tree/dev/tt_metal 

lm-evaluation-harness fork: https://github.com/tstescoTT/lm-evaluation-harness

## Step 1:  Build Docker Image

The script `build_docker.sh` handles building images for different configurations:

```bash
cd tt-inference-server/vllm-tt-metal
chmod +x build_docker.sh
./build_docker.sh --build
# or change the ubuntu version to 22.04
./build_docker.sh --build --ubuntu-version 22.04
```

### push image (only for admin deployment to GHCR)

```bash
./build_docker.sh --build --push
```

## Step 2: Run container for LM evals development

note: this requires running `setup.sh` to set up the weights for a particular model, in this example `Llama-3.3-70B-Instruct`.

```bash
cd tt-inference-server
export MODEL_NAME=Llama-3.3-70B-Instruct
export MODEL_VOLUME=$PWD/persistent_volume/volume_id_tt-metal-${MODEL_NAME}-v0.0.1/

docker run \
  --rm \
  -it \
  --env-file persistent_volume/model_envs/${MODEL_NAME}.env \
  --cap-add ALL \
  --device /dev/tenstorrent:/dev/tenstorrent \
  --volume /dev/hugepages-1G:/dev/hugepages-1G:rw \
  --volume ${MODEL_VOLUME?ERROR env var MODEL_VOLUME must be set}:/home/container_app_user/cache_root:rw \
  --shm-size 32G \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-20.04-amd64:v0.0.4-{TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG} bash
```

additionally for development you can mount the volumes:
```bash
  --volume $PWD/../vllm:/home/container_app_user/vllm \
  --volume $PWD/../lm-evaluation-harness:/home/container_app_user/lm-evaluation-harness \
```

## Step 3: Inside container setup and run vLLM

#### Option 1: use default installation in docker image

Already built into Docker image, continue to run vLLM.

#### option 2: install vLLM from github

```bash
# option 2: install from github
cd /home/container_app_user/vllm
git fetch
git checkout <branch>
git pull
pip install -e .
echo "done vllm install."
```

#### option 3: install edittable (for development) - mount from outside container

```bash
# option 3: install edittable (for development) - mount from outside container
cd /home/container_app_user/vllm
pip install -e .
echo "done vllm install."
```

#### Run vllm serving openai compatible API server

```bash
# run vllm serving
cd ~/app/src
python run_vllm_api_server.py
```

## Step 4: Sending requests to vLLM inference server

If the container is exposing a port (e.g. `docker run ... --publish 7000:7000`), you can send requests to that port , otherwise you can enter an interactive shell within the container via:
```bash
docker exec --user 1000 -it $(docker ps -q | head -n1) bash
# once in shell
cd ~/app/src
python example_requests_client.py 
```

# Reference

##  Build Docker Image Manually

When building, update the commit SHA and get correct SHA from model developers or from vLLM readme (https://github.com/tenstorrent/vllm/tree/dev/tt_metal#vllm-and-tt-metal-branches ). The Dockerfile version updates infrequently but may also be updated.
```bash
# set build context to repo root
cd tt-inference-server
# build image
export UBUNTU_VERSION="20.04"
export OS_VERSION="ubuntu-${UBUNTU_VERSION}-amd64"
export TT_METAL_DOCKERFILE_URL=ghcr.io/tenstorrent/tt-metal/tt-metalium-${OS_VERSION}-release:v0.55.0
export TT_METAL_COMMIT_SHA_OR_TAG=v0.56.0-rc39
export TT_METAL_COMMIT_DOCKER_TAG=${TT_METAL_COMMIT_SHA_OR_TAG:0:12}
export TT_VLLM_COMMIT_SHA_OR_TAG=3429acf14e46436948db6865b90178c6375d0217
export TT_VLLM_COMMIT_DOCKER_TAG=${TT_VLLM_COMMIT_SHA_OR_TAG:0:12}
export CONTAINER_APP_UID=1000
export IMAGE_VERSION=$(cat VERSION)
# build cloud deploy image
docker build \
  -t ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-cloud-${OS_VERSION}:${IMAGE_VERSION}-${TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG} \
  --build-arg TT_METAL_DOCKERFILE_URL=${TT_METAL_DOCKERFILE_URL} \
  --build-arg TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
  --build-arg TT_VLLM_COMMIT_SHA_OR_TAG=${TT_VLLM_COMMIT_SHA_OR_TAG} \
  --build-arg CONTAINER_APP_UID=${CONTAINER_APP_UID} \
  . -f vllm-tt-metal/vllm.tt-metal.src.cloud.Dockerfile
# build dev image
docker build \
  -t ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-${OS_VERSION}:${IMAGE_VERSION}-${TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG} \
  --build-arg CLOUD_DOCKERFILE_URL=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-cloud-${OS_VERSION}:${IMAGE_VERSION}-${TT_METAL_COMMIT_DOCKER_TAG}-${TT_VLLM_COMMIT_DOCKER_TAG} \
  . -f vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile
```

### Build tt-metal Ubuntu 22.04 base image manually

In the tt-metal repo there is a Ubuntu 22.04 Dockerfile: https://github.com/tenstorrent/tt-metal/blob/main/dockerfile/ubuntu-22.04-amd64.Dockerfile
This Dockerfile installs the python dependencies for Ubuntu 22.04 running Python 3.10: https://github.com/tenstorrent/tt-metal/blob/main/scripts/docker/requirements-22.04.txt

The Ubuntu 22.04 images are not yet published to GHCR as the Ubuntu 20.04 images are (https://github.com/tenstorrent/tt-metal/pkgs/container/tt-metal%2Ftt-metalium-ubuntu-20.04-amd64-release%2Fwormhole_b0)

You can build local tt-metal ubuntu 22.04 base image:
```bash
git clone --depth 1 https://github.com/tenstorrent/tt-metal.git
cd tt-metal
git fetch --depth 1 origin ${TT_METAL_COMMIT_SHA_OR_TAG}
git checkout ${TT_METAL_COMMIT_SHA_OR_TAG}
docker build \
  -t local/tt-metal/tt-metalium/${OS_VERSION}:${TT_METAL_COMMIT_SHA_OR_TAG} \
  --build-arg UBUNTU_VERSION=${UBUNTU_VERSION} \
  --target ci-build \
  -f dockerfile/Dockerfile .
export TT_METAL_DOCKERFILE_URL=local/tt-metal/tt-metalium/${OS_VERSION}:${TT_METAL_COMMIT_SHA_OR_TAG}
```

You can then repeat the steps above to build with, e.g. `TT_METAL_DOCKERFILE_URL=local/tt-metal/tt-metalium/ubuntu-22.04-amd64:latest`

