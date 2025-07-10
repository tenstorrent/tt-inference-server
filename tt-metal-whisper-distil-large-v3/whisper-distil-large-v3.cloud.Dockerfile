# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# default base image, override with --build-arg TT_METAL_DOCKERFILE_VERSION=<version>
ARG TT_METAL_DOCKERFILE_VERSION=0.0.5-v0.60.0-rc4-1d68fc6

FROM ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-cloud-ubuntu-22.04-amd64:0.0.5-v0.60.0-rc4-1d68fc6

# Build stage
LABEL maintainer="Ben Goel <bgoel@tenstorrent.com>"
# connect Github repo with package
LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

ARG DEBIAN_FRONTEND=noninteractive
# default commit sha, override with --build-arg TT_METAL_COMMIT_SHA_OR_TAG=<sha>

# make build commit SHA available in the image for reference and debugging
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG}
ENV SHELL=/bin/bash
ENV TZ=America/Los_Angeles
# tt-metal build vars
ENV ARCH_NAME=blackhole
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
    # required
    gosu \
    python3-venv \
    # system deps
    libsndfile1 \
    wget \
    nano \
    acl \
    jq \
    vim \
    git-lfs \
    # user deps
    htop \
    screen \
    tmux \
    unzip \
    zip \
    curl \
    iputils-ping \
    rsync

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
# CONTAINER_APP_UID is a random ID, change this and rebuild if it collides with host
ARG CONTAINER_APP_UID=1000
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

WORKDIR ${HOME_DIR}

# install inference server requirements
ARG APP_DIR="${HOME_DIR}/app"
ENV APP_DIR=${APP_DIR}
WORKDIR ${APP_DIR}
ENV PYTHONPATH=${PYTHONPATH}:${APP_DIR}
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "tt-metal-whisper-distil-large-v3/server" "${APP_DIR}/server"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "utils" "${APP_DIR}/utils"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "tt-metal-whisper-distil-large-v3/requirements.txt" "${APP_DIR}/requirements.txt"
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && pip install --default-timeout=240 --no-cache-dir -r requirements.txt"

# spinup inference server
WORKDIR "${TT_METAL_HOME}"
CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && pytest ${APP_DIR}/server/gunicorn_app.py -s"]
