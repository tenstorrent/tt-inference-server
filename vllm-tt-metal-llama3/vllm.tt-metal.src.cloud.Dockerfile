# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# default base image, override with --build-arg TT_METAL_DOCKERFILE_URL=<url or local image path>
# NOTE: tt-metal Ubuntu 22.04 Dockerfile must be built locally until release images are published
ARG TT_METAL_DOCKERFILE_URL

FROM ${TT_METAL_DOCKERFILE_URL}

# shared build stage, FROM is set by the OS specific Dockerfiles
LABEL maintainer="Tom Stesco <tstesco@tenstorrent.com>"
# connect Github repo with package
LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

# must set commit SHAs
ARG TT_METAL_COMMIT_SHA_OR_TAG
ARG TT_VLLM_COMMIT_SHA_OR_TAG

# CONTAINER_APP_UID is a random ID, change this and rebuild if it collides with host
ARG CONTAINER_APP_UID=15863
ARG DEBIAN_FRONTEND=noninteractive
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

# apt-get might have an outdated keyring, preventing it to download packages, so we fetch the latest
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null

# extra system deps
RUN apt-get update && apt-get install -y \
    # required
    gosu \
    # extra tt-metal TODO: remove as non longer needed
    python3-venv \
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

# build tt-metal
RUN git clone https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && git submodule update --init --recursive \
    && bash ./build_metal.sh \
    && bash ./create_venv.sh

# user setup
ENV CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}
RUN useradd -u ${CONTAINER_APP_UID} -s /bin/bash -d ${HOME_DIR} ${CONTAINER_APP_USERNAME} \
    && mkdir -p ${HOME_DIR} \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR} \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${TT_METAL_HOME}
  
USER ${CONTAINER_APP_USERNAME}

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

# extra vllm and model dependencies
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && pip install compressed-tensors \
    && pip install -r /tt-metal/models/demos/llama3/requirements.txt"

ARG APP_DIR="${HOME_DIR}/app"
WORKDIR ${APP_DIR}
ENV PYTHONPATH=${PYTHONPATH}:${APP_DIR}
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "vllm-tt-metal-llama3/src" "${APP_DIR}/src"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "vllm-tt-metal-llama3/requirements.txt" "${APP_DIR}/requirements.txt"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "utils" "${APP_DIR}/utils"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "benchmarking" "${APP_DIR}/benchmarking"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "evals" "${APP_DIR}/evals"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "tests" "${APP_DIR}/tests"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "locust" "${APP_DIR}/locust"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "VERSION" "${APP_DIR}/VERSION"
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
&& pip install --default-timeout=240 --no-cache-dir -r requirements.txt"

WORKDIR "${APP_DIR}/src"

CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && python run_vllm_api_server.py"]
