# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# default base image, override with --build-arg TT_METAL_DOCKERFILE_VERSION=<version>
ARG TT_METAL_DOCKERFILE_VERSION=v0.53.0-rc16

FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64:$TT_METAL_DOCKERFILE_VERSION-dev

# Build stage
LABEL maintainer="Tom Stesco <tstesco@tenstorrent.com>"
# connect Github repo with package
LABEL org.opencontainers.image.source https://github.com/tenstorrent/tt-inference-server

ARG DEBIAN_FRONTEND=noninteractive
# default commit sha, override with --build-arg TT_METAL_COMMIT_SHA_OR_TAG=<sha>
ARG TT_METAL_COMMIT_SHA_OR_TAG=ebdffa93d911ebf18e1fd4058a6f65ed0dff09ef
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
    patchelf \
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
    # syseng tools
    cargo \
    && rm -rf /var/lib/apt/lists/*

# build tt-metal
RUN git clone https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && git submodule update --init --recursive \
    && git submodule foreach 'git lfs fetch --all && git lfs pull' \
    && bash ./build_metal.sh \
    && bash ./create_venv.sh

# user setup
ARG HOME_DIR=/home/user
RUN useradd -u 1000 -s /bin/bash -d ${HOME_DIR} user \
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
ENV PYTHONPATH=${TT_METAL_HOME}:${vllm_dir}
ENV VLLM_TARGET_DEVICE="tt"
RUN git clone https://github.com/tenstorrent/vllm.git ${vllm_dir}\
    && cd ${vllm_dir} && git checkout ${TT_VLLM_COMMIT_SHA_OR_TAG} \
    && /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate && pip install -e ."

# extra vllm dependencies
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate && pip install compressed-tensors"

WORKDIR ${vllm_dir}
