# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# default base image, override with --build-arg TT_METAL_DOCKERFILE_URL=<url or local image path>
# NOTE: tt-metal Ubuntu 22.04 Dockerfile must be built locally until release images are published
ARG TT_METAL_DOCKERFILE_URL=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64:v0.53.0-rc34-dev

FROM ${TT_METAL_DOCKERFILE_URL}

# Build stage
LABEL maintainer="Tom Stesco <tstesco@tenstorrent.com>"
# connect Github repo with package
LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

ARG DEBIAN_FRONTEND=noninteractive
# default commit sha, override with --build-arg TT_METAL_COMMIT_SHA_OR_TAG=<sha>
ARG TT_METAL_COMMIT_SHA_OR_TAG
ARG TT_VLLM_COMMIT_SHA_OR_TAG=dev

# make build commit SHA available in the image for reference and debugging
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG}
ENV SHELL=/bin/bash
ENV TZ=America/Los_Angeles
# tt-metal build vars
ENV ARCH_NAME=wormhole_b0
ENV TT_METAL_HOME=/tt-metal
ENV CONFIG=Release
ENV TT_METAL_ENV=dev
ENV LOGURU_LEVEL=INFO
# derived vars
ENV PYTHONPATH=${TT_METAL_HOME}
# note: PYTHON_ENV_DIR is used by create_venv.sh
ENV PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env
ENV LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib

# extra system deps
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    wget \
    nano \
    acl \
    jq \
    vim \
    # user deps
    htop \
    screen \
    tmux \
    unzip \
    zip \
    curl \
    iputils-ping \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# build tt-metal (venv only for mock)
RUN git clone --depth 1 https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git fetch --depth 1 origin ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && git submodule update --init models/demos/t3000/llama2_70b/reference/llama \
    && bash ./create_venv.sh

# user setup
# CONTAINER_APP_UID is a random ID, change this and rebuild if it collides with host
ENV CONTAINER_APP_UID=15863
ENV CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}
RUN useradd -u ${CONTAINER_APP_UID} -s /bin/bash -d ${HOME_DIR} ${CONTAINER_APP_USERNAME} \
    && mkdir -p ${HOME_DIR} \
    && chown -R user:user ${HOME_DIR} \
    && chown -R user:user ${TT_METAL_HOME}
  
USER user

# tt-metal python env default
RUN echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc

# install tt-smi
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && pip3 install --upgrade pip \
    && pip3 install git+https://github.com/tenstorrent/tt-smi"

# runtime required for tt-metal on WH
ENV WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

WORKDIR ${HOME_DIR}
# vllm install, see: https://github.com/tenstorrent/vllm/blob/dev/tt_metal/README.md
ENV vllm_dir=${HOME_DIR}/vllm
ENV PYTHONPATH=${PYTHONPATH}:${vllm_dir}
ENV VLLM_TARGET_DEVICE="tt"
RUN git clone https://github.com/tenstorrent/vllm.git ${vllm_dir}\
    && cd ${vllm_dir} && git checkout ${TT_VLLM_COMMIT_SHA_OR_TAG} \
    && /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate && pip install -e ."

# extra vllm dependencies
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate && pip install compressed-tensors"

ARG APP_DIR="${HOME_DIR}/app"
WORKDIR ${APP_DIR}
ENV PYTHONPATH=${PYTHONPATH}:${APP_DIR}
COPY --chown=user:user "vllm-tt-metal-llama3/src" "${APP_DIR}/src"
COPY --chown=user:user "vllm-tt-metal-llama3/requirements.txt" "${APP_DIR}/requirements.txt"
COPY --chown=user:user "utils" "${APP_DIR}/utils"
COPY --chown=user:user "tests" "${APP_DIR}/tests"
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
&& pip install --default-timeout=240 --no-cache-dir -r requirements.txt"

WORKDIR "${APP_DIR}/tests"
CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && python mock_vllm_api_server.py"]

# Default environment variables for the Llama-3.1-70b-instruct inference server
# Note: LLAMA3_CKPT_DIR and similar variables get set by mock_vllm_api_server.py
ENV CACHE_ROOT=/home/container_app_user/cache_root
ENV HF_HOME=/home/container_app_user/cache_root/huggingface
ENV MODEL_WEIGHTS_ID=id_repacked-Llama-3.1-70B-Instruct
ENV MODEL_WEIGHTS_PATH=/home/container_app_user/cache_root/model_weights/repacked-Llama-3.1-70B-Instruct
ENV LLAMA_VERSION=llama3
ENV SERVICE_PORT=7000
