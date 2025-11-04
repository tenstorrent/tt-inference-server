# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

FROM ghcr.io/tenstorrent/tt-xla/tt-xla-ird-ubuntu-22-04:latest

# Build stage
LABEL maintainer="Igor Djuric <idjuric@tenstorrent.com>"
# connect Github repo with package
LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

ARG DEBIAN_FRONTEND=noninteractive
# default commit sha, override with --build-arg TT_XLA_COMMIT_SHA_OR_TAG=<sha>
ARG TT_XLA_COMMIT_SHA_OR_TAG=85ef39dbf090e26e018b21b78c1694ae370b63cd


# CONTAINER_APP_UID is a random ID, change this and rebuild if it collides with host
ARG CONTAINER_APP_UID=1000
ARG DEBIAN_FRONTEND=noninteractive

# make build commit SHA available in the image for reference and debugging
ENV TT_XLA_COMMIT_SHA_OR_TAG=${TT_XLA_COMMIT_SHA_OR_TAG}

ENV LOG_LEVEL=INFO
ENV ENVIRONMENT=development
ENV DEVICE_IDS=0
# max queue size 32 x 2 for Galaxy
ENV MAX_MODEL_LENGTH=64
ENV MAX_NUM_BATCHED_TOKENS=64
ENV MAX_NUM_SEQS=1

ENV MAX_QUEUE_SIZE=64
ENV MAX_BATCH_SIZE=1
ENV MODEL_RUNNER=${MODEL_RUNNER}
ENV NUM_INFERENCE_STEPS=20
ENV TT_MM_THROTTLE_PERF=5

ENV SHELL=/bin/bash
ENV CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}
ENV TT_XLA_HOME=${HOME_DIR}/tt-xla
ENV TTPJRT_SOURCE_DIR=${TT_XLA_HOME}
# tt-metal build vars
ENV TT_METAL_HOME=${HOME_DIR}/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal
ENV CONFIG=Release
ENV TT_METAL_ENV=dev
ENV LOGURU_LEVEL=INFO
# derived vars
ENV PYTHONPATH=${HOME_DIR}/tt-xla:/${HOME_DIR}/tt-xla/tests:/usr/local/lib/python3.11/dist-packages:/${HOME_DIR}/tt-xla/integrations/vllm_plugin:/${HOME_DIR}/tt-xla/third_party/tt-mlir/src/tt-mlir/build/python_packages

# apt-get might have an outdated keyring, preventing it to download packages, so we fetch the latest
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null


# extra system deps
RUN apt-get update && apt-get install -y \
    # required
    software-properties-common \
    gosu \
    libgl1 \
    libsndfile1 \
    wget \
    git \
    git-lfs \
    nano \
    acl \
    jq \
    vim \
    python3-pip \
    python3-venv \
    # audio processing
    ffmpeg \
    # pyluwen build dependencies (Rust package with protobuf)
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    # build tools
    ninja-build \
    ccache \
    libhwloc-dev \
    pandoc \
    libtbb-dev \
    libcapstone-dev \
    linux-tools-generic \
    libgtest-dev \
    doxygen \
    graphviz \
    patchelf \
    libyaml-cpp-dev \
    libboost-all-dev \
    lcov \
    xxd \
    # utilities
    htop \
    screen \
    tmux \
    unzip \
    zip \
    curl \
    iputils-ping \
    rsync \
    ssh \
    sudo \
    tmux \
    psmisc \
    bash-completion \
    && rm -rf /var/lib/apt/lists/* \
    && protoc --version

RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.29.2/cmake-3.29.2-linux-x86_64.sh -o cmake.sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh && \
    cmake --version

# user setup
RUN useradd -u ${CONTAINER_APP_UID} -s /bin/bash -d ${HOME_DIR} ${CONTAINER_APP_USERNAME} \
    && mkdir -p ${HOME_DIR} \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR} \
    && mkdir -p /run/sshd \
    && chmod 755 /run/sshd

USER ${CONTAINER_APP_USERNAME}

# build tt-xla with TT-MLIR toolchain
RUN git clone https://github.com/tenstorrent/tt-xla.git ${TT_XLA_HOME} \
    && cd ${TT_XLA_HOME} \
    && git checkout ${TT_XLA_COMMIT_SHA_OR_TAG} \
    && git submodule update --init --recursive \
    && export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain \
    && source ${TT_XLA_HOME}/venv/activate \
    && pip install -r ${TT_XLA_HOME}/python_package/requirements.txt --no-cache-dir \
    && pip install -r ${TT_XLA_HOME}/venv/requirements-dev.txt --no-cache-dir \
    && echo "=== Installing VLLM with ignore-installed ===" \
    && pip install -r integrations/vllm_plugin/requirements-vllm-plugin.txt --no-cache-dir --ignore-installed \
    && pip install -e integrations/vllm_plugin --ignore-installed \
    && python -c "import vllm; print('VLLM installed successfully')" \
    && cmake -G Ninja -B build \
    && cmake --build build

RUN echo "source ${TT_XLA_HOME}/venv/activate" >> ${HOME_DIR}/.bashrc

# install tt-smi
RUN ${TT_XLA_HOME}/venv/bin/pip install --upgrade pip \
    && export CARGO_HOME=${HOME_DIR}/.cargo \
    && export RUSTUP_HOME=${HOME_DIR}/.rustup \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path \
    && . ${HOME_DIR}/.cargo/env \
    && rustup update \
    && ${TT_XLA_HOME}/venv/bin/pip install git+https://github.com/tenstorrent/tt-smi
RUN echo "source ${TT_XLA_HOME}/venv/activate" >> ${HOME_DIR}/.bashrc

WORKDIR ${HOME_DIR}

# install inference server requirements
ARG APP_DIR="${HOME_DIR}/app"
ENV APP_DIR=${APP_DIR}
WORKDIR ${APP_DIR}
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "tt-media-server" "${APP_DIR}/server"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "tt-media-server/requirements.txt" "${APP_DIR}/requirements.txt"
RUN cd ${APP_DIR} && ${TT_XLA_HOME}/venv/bin/pip install --default-timeout=240 --no-cache-dir -r requirements.txt || \
    (echo 'Requirements installation failed, trying without pyluwen...' && \
     grep -v pyluwen requirements.txt > requirements_no_pyluwen.txt && \
     ${TT_XLA_HOME}/venv/bin/pip install --default-timeout=240 --no-cache-dir -r requirements_no_pyluwen.txt)

# spinup inference server
WORKDIR "${APP_DIR}"

EXPOSE 8000
CMD ["/bin/bash", "-c", "cd ${APP_DIR}/server/ && ${TT_XLA_HOME}/venv/bin/python -m uvicorn --host 0.0.0.0 main:app --lifespan on --port 8000"]
