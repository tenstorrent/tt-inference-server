# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

ARG MODEL_SERVICE=image

#FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64:$TT_METAL_DOCKERFILE_VERSION-dev
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest AS runtime

# Build stage
LABEL maintainer="Igor Djuric <idjuric@tenstorrent.com>"
# connect Github repo with package
LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

ARG DEBIAN_FRONTEND=noninteractive
# default commit sha, override with --build-arg TT_METAL_COMMIT_SHA_OR_TAG=<sha>
ARG TT_METAL_COMMIT_SHA_OR_TAG=88bc225cb4555d20f9af3ec3d959893066bb3254

# default model runner, override with --build-arg MODEL_RUNNER=<runner>
ARG MODEL_RUNNER=tt-sdxl

# CONTAINER_APP_UID is a random ID, change this and rebuild if it collides with host
ARG CONTAINER_APP_UID=1000
ARG DEBIAN_FRONTEND=noninteractive

# make build commit SHA available in the image for reference and debugging
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG}

ENV MODEL_SERVICE=image
ENV LOG_LEVEL=INFO
ENV ENVIRONMENT=development
ENV DEVICE_IDS=0
# max queue size 32 x 2 for Galayx
ENV MAX_QUEUE_SIZE=64
ENV MAX_BATCH_SIZE=32
ENV MODEL_RUNNER=${MODEL_RUNNER}
ENV NUM_INFERENCE_STEPS=20
ENV TT_MM_THROTTLE_PERF=5

ENV SHELL=/bin/bash
ENV CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}
# tt-metal build vars
ENV TT_METAL_HOME=${HOME_DIR}/tt-metal
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
    libgl1 \
    libsndfile1 \
    wget \
    git-lfs \
    nano \
    acl \
    jq \
    vim \
    python3-pip \
    # pyluwen build dependencies (Rust package with protobuf)
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    htop \
    screen \
    tmux \
    unzip \
    zip \
    curl \
    iputils-ping \
    rsync \
    && rm -rf /var/lib/apt/lists/* \
    && protoc --version

RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh -o cmake.sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh && \
    cmake --version
    
# build tt-metal
RUN git clone https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && git submodule update --init --recursive \
    && ./install_dependencies.sh --docker \
    && bash ./create_venv.sh \
    && echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc \
    && pip3 install --upgrade setuptools wheel \
    && bash ./build_metal.sh --release -e \
    && pip3 install cmake \
    && pip3 install cmake --upgrade \
    && hash -r

# user setup
RUN useradd -u ${CONTAINER_APP_UID} -s /bin/bash -d ${HOME_DIR} ${CONTAINER_APP_USERNAME} \
    && mkdir -p ${HOME_DIR} \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR} \
    && mkdir -p /run/sshd \
    && chmod 755 /run/sshd

USER ${CONTAINER_APP_USERNAME}

RUN echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc

# install tt-smi
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && pip3 install --upgrade pip \
    && export CARGO_HOME=${HOME_DIR}/.cargo \
    && export RUSTUP_HOME=${HOME_DIR}/.rustup \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path \
    && . ${HOME_DIR}/.cargo/env \
    && rustup update \
    && pip3 install git+https://github.com/tenstorrent/tt-smi"

WORKDIR ${HOME_DIR}

# install inference server requirements
ARG APP_DIR="${HOME_DIR}/app"
ENV APP_DIR=${APP_DIR}
WORKDIR ${APP_DIR}
ENV PYTHONPATH=${TT_METAL_HOME}
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "utils" "${APP_DIR}/utils"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "benchmarking" "${APP_DIR}/benchmarking"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "evals" "${APP_DIR}/evals"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "tests" "${APP_DIR}/tests"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "locust" "${APP_DIR}/locust"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "VERSION" "${APP_DIR}/VERSION"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "tt-metal-sdxl" "${TT_METAL_HOME}/server"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "tt-metal-sdxl/requirements.txt" "${TT_METAL_HOME}/requirements.txt"
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && cd ${TT_METAL_HOME} && pip install --default-timeout=240 --no-cache-dir -r requirements.txt || \
    (echo 'Requirements installation failed, trying without pyluwen...' && \
     grep -v pyluwen requirements.txt > requirements_no_pyluwen.txt && \
     pip install --default-timeout=240 --no-cache-dir -r requirements_no_pyluwen.txt)"

# spinup inference server
WORKDIR "${TT_METAL_HOME}"
ENTRYPOINT []
EXPOSE 8000
CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && cd ${TT_METAL_HOME}/server/ &&  source ./run_uvicorn.sh"]


FROM runtime as forge_runtime

COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} --chmod=755 "tt-metal-sdxl/forge_entrypoint.sh" "${HOME_DIR}/forge_entrypoint.sh"

WORKDIR ${TT_METAL_HOME}

# Install forge-specific requirements
RUN /bin/bash -c "cd ${TT_METAL_HOME}/server && \
    python3 -m venv venv-worker && \
    source venv-worker/bin/activate && \
    pip install -r requirements.txt && \
    pip install -r tt_model_runners/forge_runners/requirements.txt && \
    pip cache purge"

ENTRYPOINT ["/home/container_app_user/forge_entrypoint.sh"]
CMD ["/bin/bash", "-c", "source venv-worker/bin/activate && cd ./server/ &&  source ./run_uvicorn.sh"]