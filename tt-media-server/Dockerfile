# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# ============================================================================
# Build stage - contains all build tools and artifacts
# ============================================================================
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest AS builder

LABEL maintainer="Igor Djuric <idjuric@tenstorrent.com>"
LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

ARG DEBIAN_FRONTEND=noninteractive
ARG TT_METAL_COMMIT_SHA_OR_TAG=0d65e030a976f5bc33f22d22d5adfb7736a9e884
ARG CONTAINER_APP_UID=1000
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}

ENV TT_METAL_HOME=${HOME_DIR}/tt-metal
ENV CONFIG=Release
ENV TT_METAL_ENV=dev
ENV PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env
ENV LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib
ENV PYTHONPATH=${TT_METAL_HOME}

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    libffi-dev \
    libssl-dev \
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    wget \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install CMake
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh -o cmake.sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh

# Build tt-metal
RUN git clone --depth 1 --single-branch https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} && \
    cd ${TT_METAL_HOME} && \
    git fetch --depth 1 origin ${TT_METAL_COMMIT_SHA_OR_TAG} && \
    git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} && \
    git submodule update --init --recursive && \
    ./install_dependencies.sh --docker && \
    bash ./create_venv.sh && \
    bash -c "source ${PYTHON_ENV_DIR}/bin/activate && pip3 install --upgrade setuptools wheel cmake" && \
    bash ./build_metal.sh --release -e && \
    rm -rf ${TT_METAL_HOME}/.git && \
    find ${TT_METAL_HOME} -name "*.git" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find ${PYTHON_ENV_DIR} -name "*.pyc" -delete && \
    find ${PYTHON_ENV_DIR} -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Install tt-smi
RUN bash -c "source ${PYTHON_ENV_DIR}/bin/activate && \
    export CARGO_HOME=${HOME_DIR}/.cargo && \
    export RUSTUP_HOME=${HOME_DIR}/.rustup && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path && \
    . ${HOME_DIR}/.cargo/env && \
    pip3 install git+https://github.com/tenstorrent/tt-smi && \
    rm -rf ${HOME_DIR}/.cargo/registry ${HOME_DIR}/.cargo/git ${HOME_DIR}/.rustup/toolchains/*/share"

# Copy application files and install requirements
COPY "tt-media-server" "${TT_METAL_HOME}/server"
COPY "tt-media-server/requirements.txt" "${TT_METAL_HOME}/requirements.txt"
COPY "utils" "${TT_METAL_HOME}/utils"
COPY "utils/sdxl_accuracy" "${TT_METAL_HOME}/sdxl_accuracy"

RUN bash -c "source ${PYTHON_ENV_DIR}/bin/activate && \
    cd ${TT_METAL_HOME} && \
    pip install --default-timeout=240 --no-cache-dir -r requirements.txt || \
    (echo 'Requirements installation failed, trying without pyluwen...' && \
     grep -v pyluwen requirements.txt > requirements_no_pyluwen.txt && \
     pip install --default-timeout=240 --no-cache-dir -r requirements_no_pyluwen.txt) && \
    find ${PYTHON_ENV_DIR} -name "*.pyc" -delete && \
    find ${PYTHON_ENV_DIR} -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true"

# ============================================================================
# Runtime stage - minimal image with only runtime dependencies
# ============================================================================
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest

LABEL maintainer="Igor Djuric <idjuric@tenstorrent.com>"
LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

ARG DEBIAN_FRONTEND=noninteractive
ARG TT_METAL_COMMIT_SHA_OR_TAG=0d65e030a976f5bc33f22d22d5adfb7736a9e884
ARG CONTAINER_APP_UID=1000

ENV CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}

# Runtime environment variables
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG}
ENV LOG_LEVEL=INFO
ENV ENVIRONMENT=development
ENV DEVICE_IDS=0
ENV MAX_QUEUE_SIZE=64
ENV MAX_BATCH_SIZE=1
ENV MODEL_RUNNER=${MODEL_RUNNER}
ENV NUM_INFERENCE_STEPS=20
ENV TT_MM_THROTTLE_PERF=5
ENV SHELL=/bin/bash
ENV TT_METAL_HOME=${HOME_DIR}/tt-metal
ENV CONFIG=Release
ENV TT_METAL_ENV=dev
ENV LOGURU_LEVEL=INFO
ENV PYTHONPATH=${TT_METAL_HOME}
ENV PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env
ENV LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib

# Install runtime dependencies + tools needed for runtime pip installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    libgl1 \
    libsndfile1 \
    libprotobuf23 \
    libffi8 \
    libssl3 \
    python3 \
    python3-pip \
    python3-dev \
    python3-distutils \
    ca-certificates \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create user
RUN useradd -u ${CONTAINER_APP_UID} -s /bin/bash -d ${HOME_DIR} ${CONTAINER_APP_USERNAME} && \
    mkdir -p ${HOME_DIR} /run/sshd && \
    chmod 755 /run/sshd

# Copy tt-metal from builder
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${TT_METAL_HOME} ${TT_METAL_HOME}
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR}/.cargo ${HOME_DIR}/.cargo

# Set ownership
RUN chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR}

USER ${CONTAINER_APP_USERNAME}

RUN echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc

WORKDIR ${TT_METAL_HOME}

EXPOSE 8000
CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && cd ${TT_METAL_HOME}/server/ && source ./run_uvicorn.sh"]
