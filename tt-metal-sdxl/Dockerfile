# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

ARG MODEL_SERVICE=image

# ============================================================================
# Builder Stage - Contains all build dependencies and compilation
# ============================================================================
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG TT_METAL_COMMIT_SHA_OR_TAG=88bc225cb4555d20f9af3ec3d959893066bb3254
ARG MODEL_RUNNER=tt-sdxl
ARG CONTAINER_APP_UID=1000
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}

# Environment variables for build
ENV TT_METAL_HOME=${HOME_DIR}/tt-metal
ENV CONFIG=Release
ENV TT_METAL_ENV=dev
ENV PYTHONPATH=${TT_METAL_HOME}
ENV PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env
ENV LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib
ENV SHELL=/bin/bash

# Install build dependencies and clean up in same layer
RUN apt-get update && apt-get install -y \
    # Build dependencies
    build-essential \
    cmake \
    python3-dev \
    python3-pip \
    libffi-dev \
    libssl-dev \
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    git-lfs \
    wget \
    curl \
    unzip \
    # Additional dependencies for Rust/tt-smi
    ca-certificates \
    # Runtime dependencies (will be copied to final stage)
    gosu \
    libgl1 \
    libsndfile1 \
    # Add sudo for dependency installation
    sudo \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install newer CMake
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh -o cmake.sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh

# Clone tt-metal as root first to install dependencies
RUN git clone https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} && \
    cd ${TT_METAL_HOME} && \
    git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} && \
    git submodule update --init --recursive && \
    echo "Installing tt-metal dependencies as root..." && \
    ./install_dependencies.sh --docker

# Create user more robustly - handle existing UID gracefully
RUN set -e && \
    # Check if UID already exists and remove if necessary
    if id ${CONTAINER_APP_UID} >/dev/null 2>&1; then \
        existing_user=$(id -nu ${CONTAINER_APP_UID}); \
        echo "UID ${CONTAINER_APP_UID} already exists for user $existing_user, removing..."; \
        userdel -r "$existing_user" 2>/dev/null || true; \
    fi && \
    # Create group first
    groupadd -g ${CONTAINER_APP_UID} ${CONTAINER_APP_USERNAME} 2>/dev/null || true && \
    # Create user with explicit flags
    useradd \
        --uid ${CONTAINER_APP_UID} \
        --gid ${CONTAINER_APP_UID} \
        --shell /bin/bash \
        --home-dir ${HOME_DIR} \
        --create-home \
        --no-log-init \
        ${CONTAINER_APP_USERNAME} && \
    # Ensure proper ownership of tt-metal directory
    chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR} ${TT_METAL_HOME}

USER ${CONTAINER_APP_USERNAME}
WORKDIR ${HOME_DIR}

# Now build tt-metal as the user (dependencies already installed)
RUN set -e && \
    cd ${TT_METAL_HOME} && \
    echo "Creating virtual environment..." && \
    bash ./create_venv.sh && \
    echo "Building tt-metal..." && \
    bash ./build_metal.sh --release -e && \
    echo "Cleaning up build artifacts..." && \
    # Clean up git history and build artifacts to save space
    rm -rf .git && \
    find . -name "*.o" -delete && \
    find . -name "*.a" -delete && \
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    echo "tt-metal build completed successfully"

# Verify Python environment exists and upgrade pip
RUN set -e && \
    echo "Checking Python environment..." && \
    ls -la ${PYTHON_ENV_DIR}/bin/ && \
    if [ -f "${PYTHON_ENV_DIR}/bin/activate" ]; then \
        echo "Python environment found, upgrading pip..." && \
        /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate && pip3 install --upgrade pip setuptools wheel"; \
    else \
        echo "ERROR: Python environment not found at ${PYTHON_ENV_DIR}" && \
        exit 1; \
    fi

# Install tt-smi (optional - will continue if it fails)
RUN set -e && \
    echo "Installing Rust toolchain..." && \
    export CARGO_HOME=${HOME_DIR}/.cargo && \
    export RUSTUP_HOME=${HOME_DIR}/.rustup && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path && \
    echo "Rust installation completed"

RUN set -e && \
    echo "Installing tt-smi..." && \
    export CARGO_HOME=${HOME_DIR}/.cargo && \
    export RUSTUP_HOME=${HOME_DIR}/.rustup && \
    /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate && \
        . ${HOME_DIR}/.cargo/env && \
        pip3 install git+https://github.com/tenstorrent/tt-smi" || \
    echo "WARNING: tt-smi installation failed, continuing without it..."

# Clean up Rust toolchain after tt-smi installation
RUN rm -rf ${HOME_DIR}/.cargo ${HOME_DIR}/.rustup || true

# Copy and install application requirements
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "tt-metal-sdxl/requirements.txt" "${TT_METAL_HOME}/requirements.txt"
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate && \
    cd ${TT_METAL_HOME} && \
    pip install --default-timeout=240 --no-cache-dir -r requirements.txt || \
    (echo 'Requirements installation failed, trying without pyluwen...' && \
     grep -v pyluwen requirements.txt > requirements_no_pyluwen.txt && \
     pip install --default-timeout=240 --no-cache-dir -r requirements_no_pyluwen.txt) && \
    # Clean up pip cache
    pip cache purge"

# ============================================================================
# Runtime Stage - Minimal runtime environment
# ============================================================================
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest AS runtime

LABEL maintainer="Igor Djuric <idjuric@tenstorrent.com>"
LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

ARG DEBIAN_FRONTEND=noninteractive
ARG TT_METAL_COMMIT_SHA_OR_TAG=88bc225cb4555d20f9af3ec3d959893066bb3254
ARG MODEL_RUNNER=tt-sdxl
ARG CONTAINER_APP_UID=1000
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}

# Runtime environment variables
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG}
ENV MODEL_SERVICE=image
ENV LOG_LEVEL=INFO
ENV ENVIRONMENT=development
ENV DEVICE_IDS=0
ENV DEVICES_PER_RUNNER=1
ENV MAX_QUEUE_SIZE=64
ENV MAX_BATCH_SIZE=32
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

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    # Runtime essentials
    gosu \
    libgl1 \
    libsndfile1 \
    wget \
    git-lfs \
    nano \
    vim \
    python3-pip \
    # Monitoring tools (optional, could be removed)
    htop \
    # Network tools (optional)
    curl \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create runtime user with same robust approach
RUN set -e && \
    # Check if UID already exists and remove if necessary
    if id ${CONTAINER_APP_UID} >/dev/null 2>&1; then \
        existing_user=$(id -nu ${CONTAINER_APP_UID}); \
        echo "UID ${CONTAINER_APP_UID} already exists for user $existing_user, removing..."; \
        userdel -r "$existing_user" 2>/dev/null || true; \
    fi && \
    # Create group first
    groupadd -g ${CONTAINER_APP_UID} ${CONTAINER_APP_USERNAME} 2>/dev/null || true && \
    # Create user with explicit flags
    useradd \
        --uid ${CONTAINER_APP_UID} \
        --gid ${CONTAINER_APP_UID} \
        --shell /bin/bash \
        --home-dir ${HOME_DIR} \
        --create-home \
        --no-log-init \
        ${CONTAINER_APP_USERNAME} && \
    # Ensure proper ownership
    mkdir -p ${HOME_DIR}/.config && \
    chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR}

USER ${CONTAINER_APP_USERNAME}
WORKDIR ${HOME_DIR}

# Copy built artifacts from builder stage
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${TT_METAL_HOME} ${TT_METAL_HOME}

# Set up bash environment
RUN echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc

# Copy application code
ARG APP_DIR="${HOME_DIR}/app"
ENV APP_DIR=${APP_DIR}
WORKDIR ${APP_DIR}

COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "utils" "${APP_DIR}/utils"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "benchmarking" "${APP_DIR}/benchmarking"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "evals" "${APP_DIR}/evals"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "tests" "${APP_DIR}/tests"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "locust" "${APP_DIR}/locust"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "VERSION" "${APP_DIR}/VERSION"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} "tt-metal-sdxl" "${TT_METAL_HOME}/server"

# Final setup
WORKDIR "${TT_METAL_HOME}/server"
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && uvicorn --host 0.0.0.0 main:app --lifespan on --port 8000"]