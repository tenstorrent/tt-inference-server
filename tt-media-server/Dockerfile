# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# ============================================================================
# Build stage - contains all build tools and artifacts
# ============================================================================
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest AS builder

LABEL maintainer="Igor Djuric <idjuric@tenstorrent.com>" \
      org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

ARG DEBIAN_FRONTEND=noninteractive
ARG TT_METAL_COMMIT_SHA_OR_TAG=0d65e030a976f5bc33f22d22d5adfb7736a9e884
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}

ENV TT_METAL_HOME=${HOME_DIR}/tt-metal \
    CONFIG=Release \
    TT_METAL_ENV=dev \
    PYTHON_ENV_DIR=${HOME_DIR}/tt-metal/python_env \
    LD_LIBRARY_PATH=${HOME_DIR}/tt-metal/build/lib \
    PYTHONPATH=${HOME_DIR}/tt-metal

# Install build dependencies (single layer, cleaned)
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CMake (separate layer for better caching)
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh -o /tmp/cmake.sh && \
    chmod +x /tmp/cmake.sh && \
    /tmp/cmake.sh --skip-license --prefix=/usr/local && \
    rm /tmp/cmake.sh

# Build tt-metal (this is the heaviest layer - keep stable)
RUN git clone --depth 1 --single-branch https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} && \
    cd ${TT_METAL_HOME} && \
    git fetch --depth 1 origin ${TT_METAL_COMMIT_SHA_OR_TAG} && \
    git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} && \
    git submodule update --init --recursive && \
    ./install_dependencies.sh --docker && \
    bash ./create_venv.sh && \
    bash -c "source ${PYTHON_ENV_DIR}/bin/activate && pip3 install --no-cache-dir --upgrade setuptools wheel cmake" && \
    bash ./build_metal.sh --release -e && \
    # Aggressive cleanup
    rm -rf ${TT_METAL_HOME}/.git && \
    find ${TT_METAL_HOME} -type d -name ".git" -prune -exec rm -rf {} + && \
    find ${PYTHON_ENV_DIR} -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    find ${PYTHON_ENV_DIR} -type f -name "*.pyc" -delete && \
    find ${PYTHON_ENV_DIR} -type f -name "*.pyo" -delete && \
    find ${PYTHON_ENV_DIR}/lib -type d -name "tests" -prune -exec rm -rf {} + && \
    find ${PYTHON_ENV_DIR}/lib -type d -name "test" -prune -exec rm -rf {} +

# Install tt-smi (separate layer)
RUN bash -c "source ${PYTHON_ENV_DIR}/bin/activate && \
    export CARGO_HOME=${HOME_DIR}/.cargo && \
    export RUSTUP_HOME=${HOME_DIR}/.rustup && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path && \
    . ${HOME_DIR}/.cargo/env && \
    pip3 install --no-cache-dir git+https://github.com/tenstorrent/tt-smi && \
    # Cleanup Rust artifacts but keep binaries
    rm -rf ${HOME_DIR}/.cargo/registry ${HOME_DIR}/.cargo/git ${HOME_DIR}/.rustup/toolchains/*/share ${HOME_DIR}/.rustup/toolchains/*/lib"

# Copy application files (top layer - changes frequently)
COPY "tt-media-server/requirements.txt" "${TT_METAL_HOME}/requirements.txt"

# Install app requirements (separate from app code for better caching)
RUN bash -c "source ${PYTHON_ENV_DIR}/bin/activate && \
    cd ${TT_METAL_HOME} && \
    pip install --default-timeout=240 --no-cache-dir -r requirements.txt || \
    (echo 'Requirements installation failed, trying without pyluwen...' && \
     grep -v pyluwen requirements.txt > requirements_no_pyluwen.txt && \
     pip install --default-timeout=240 --no-cache-dir -r requirements_no_pyluwen.txt) && \
    find ${PYTHON_ENV_DIR} -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    find ${PYTHON_ENV_DIR} -type f -name "*.pyc" -delete"

# Copy app code (top layer - most frequently changing)
COPY "tt-media-server" "${TT_METAL_HOME}/server"
COPY "utils" "${TT_METAL_HOME}/utils"
COPY "utils/sdxl_accuracy" "${TT_METAL_HOME}/sdxl_accuracy"

# ============================================================================
# Runtime stage - minimal image with runtime dependencies
# ============================================================================
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest

LABEL maintainer="Igor Djuric <idjuric@tenstorrent.com>" \
      org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

ARG DEBIAN_FRONTEND=noninteractive
ARG TT_METAL_COMMIT_SHA_OR_TAG=0d65e030a976f5bc33f22d22d5adfb7736a9e884
ARG CONTAINER_APP_UID=1000
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}

# Runtime environment variables (single layer)
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
    LOG_LEVEL=INFO \
    ENVIRONMENT=development \
    DEVICE_IDS=0 \
    MAX_QUEUE_SIZE=64 \
    MAX_BATCH_SIZE=1 \
    MODEL_RUNNER=${MODEL_RUNNER} \
    NUM_INFERENCE_STEPS=20 \
    TT_MM_THROTTLE_PERF=5 \
    SHELL=/bin/bash \
    CONTAINER_APP_USERNAME=${CONTAINER_APP_USERNAME} \
    TT_METAL_HOME=${HOME_DIR}/tt-metal \
    CONFIG=Release \
    TT_METAL_ENV=dev \
    LOGURU_LEVEL=INFO \
    PYTHONPATH=${HOME_DIR}/tt-metal \
    PYTHON_ENV_DIR=${HOME_DIR}/tt-metal/python_env \
    LD_LIBRARY_PATH=${HOME_DIR}/tt-metal/build/lib

# Install runtime dependencies (single layer, cleaned)
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

# Create user (single layer)
RUN useradd -u ${CONTAINER_APP_UID} -s /bin/bash -m -d ${HOME_DIR} ${CONTAINER_APP_USERNAME} && \
    mkdir -p /run/sshd && \
    chmod 755 /run/sshd

# Copy tt-metal from builder (this is the big layer, but stable)
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR}/tt-metal ${HOME_DIR}/tt-metal
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR}/.cargo/bin ${HOME_DIR}/.cargo/bin

USER ${CONTAINER_APP_USERNAME}

RUN echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc

WORKDIR ${TT_METAL_HOME}

EXPOSE 8000
CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && cd ${TT_METAL_HOME}/server/ && source ./run_uvicorn.sh"]
