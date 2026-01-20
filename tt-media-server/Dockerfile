# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# ==============================================================================
# BUILDER STAGE - build tt-metal and inference server
# ==============================================================================
# Note: Dependencies installed in this stage are NOT included in the final image.
# Only the built artifacts and files are copied to the runtime stage.
# Runtime dependencies must be installed separately in the runtime stage.
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest AS builder

# Build stage
LABEL maintainer="Igor Djuric <idjuric@tenstorrent.com>"
# connect Github repo with package
LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

ARG DEBIAN_FRONTEND=noninteractive
# default commit sha, override with --build-arg TT_METAL_COMMIT_SHA_OR_TAG=<sha>
ARG TT_METAL_COMMIT_SHA_OR_TAG=6763318aa31eee19b15b8be19258e7a206396442


# CONTAINER_APP_UID is a random ID, change this and rebuild if it collides with host
ARG CONTAINER_APP_UID=1000
ARG DEBIAN_FRONTEND=noninteractive

# make build commit SHA available in the image for reference and debugging
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG}

ENV ENVIRONMENT=development
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

# Install uv for faster package management (replaces pip)
# Use system python since base image /opt/venv doesn't have pip
# tt-metal will create its own venv at ${PYTHON_ENV_DIR} later
RUN /usr/bin/python3 -m pip install --no-cache-dir uv

RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.29.2/cmake-3.29.2-linux-x86_64.sh -o cmake.sh && \
    chmod +x cmake.sh && \
    ./cmake.sh --skip-license --prefix=/usr/local && \
    rm cmake.sh && \
    cmake --version

# build tt-metal
RUN git clone --depth 1 https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git fetch --depth 1 origin ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && git submodule update --init --recursive --depth 1 \
    && ./install_dependencies.sh --docker \
    && CXX=clang++-17 CC=clang-17 bash ./create_venv.sh \
    && echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc \
    && uv pip install --upgrade setuptools wheel \
    && bash ./build_metal.sh --release -e \
    && uv pip install --upgrade cmake \
    && hash -r \
    && find ${TT_METAL_HOME} -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true

RUN mkdir -p ${HOME_DIR} \
    && mkdir -p /run/sshd \
    && chmod 755 /run/sshd

RUN echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc

# install tt-smi
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && export CARGO_HOME=${HOME_DIR}/.cargo \
    && export RUSTUP_HOME=${HOME_DIR}/.rustup \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path \
    && . ${HOME_DIR}/.cargo/env \
    && rustup update \
    && uv pip install git+https://github.com/tenstorrent/tt-smi \
    && find ${HOME_DIR} -name '.git' -type d -exec rm -rf {} + 2>/dev/null || true"

WORKDIR ${HOME_DIR}

# install inference server requirements
ARG APP_DIR="${HOME_DIR}/app"
ENV APP_DIR=${APP_DIR}
WORKDIR ${APP_DIR}
ENV PYTHONPATH=${TT_METAL_HOME}
COPY "tt-media-server" "${TT_METAL_HOME}/server"
COPY "tt-media-server/requirements.txt" "${TT_METAL_HOME}/requirements.txt"
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && cd ${TT_METAL_HOME} && uv pip install --no-cache-dir -r requirements.txt || \
    (echo 'Requirements installation failed, trying without pyluwen...' && \
    grep -v pyluwen requirements.txt > requirements_no_pyluwen.txt && \
    uv pip install --no-cache-dir -r requirements_no_pyluwen.txt)"

# Install tt-vllm-plugin
#COPY "tt-vllm-plugin" "${TT_METAL_HOME}/tt-vllm-plugin"
#RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
#    && cd ${TT_METAL_HOME}/tt-vllm-plugin \
#    && uv pip install ."

# Cleanup build-only directories
RUN rm -rf ${HOME_DIR}/.rustup \
    && rm -rf ${HOME_DIR}/.cargo \
    && rm -rf ${HOME_DIR}/.cache \
    && rm -rf ${HOME_DIR}/tt-metal/.cpmcache \
    && rm -rf ${HOME_DIR}/tt-metal/docs \
    && rm -rf ${HOME_DIR}/tt-metal/tests \
    && rm -rf ${HOME_DIR}/tt-metal/tech_reports \
    && rm -rf ${HOME_DIR}/tt-metal/.github \
    && rm -rf ${HOME_DIR}/tt-metal/contributing \
    && rm -rf ${HOME_DIR}/tt-metal/releases \
    && rm -rf ${HOME_DIR}/tt-metal/dockerfile \
    && rm -rf ${HOME_DIR}/tt-metal/cmake \
    && rm -rf ${HOME_DIR}/tt-metal/scripts \
    && rm -rf ${HOME_DIR}/tt-metal/infra/tests \
    && rm -rf ${HOME_DIR}/tt-metal/tt-train \
    && rm -rf ${HOME_DIR}/tt-metal/server/tests \
    && rm -rf ${HOME_DIR}/tt-metal/server/scripts \
    && rm -rf /usr/local/rustup \
    && rm -rf /usr/include


# ==============================================================================
# RUNTIME STAGE - final lightweight image
# ==============================================================================
# This is the final image that will be used. Only the built artifacts from the builder stage
# are copied here. Any runtime dependencies needed by the application must be installed in this stage.
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest AS runtime

LABEL maintainer="Igor Djuric <idjuric@tenstorrent.com>"
LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

# Reuse all args and environment variables
ARG DEBIAN_FRONTEND=noninteractive
ARG TT_METAL_COMMIT_SHA_OR_TAG=3e24a97c679eaf257648f8bd88486a7f9c990a6c
ARG CONTAINER_APP_UID=1000
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}

ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
    ENVIRONMENT=development \
    TT_MM_THROTTLE_PERF=5 \
    SHELL=/bin/bash \
    CONTAINER_APP_USERNAME=${CONTAINER_APP_USERNAME} \
    TT_METAL_HOME=${HOME_DIR}/tt-metal \
    CONFIG=Release \
    TT_METAL_ENV=dev \
    LOGURU_LEVEL=INFO \
    PYTHONPATH=${HOME_DIR}/tt-metal \
    PYTHON_ENV_DIR=${HOME_DIR}/tt-metal/python_env \
    LD_LIBRARY_PATH=${HOME_DIR}/tt-metal/build/lib \
    VLLM_USE_V1=1 \
    TT_METAL_LOGS_PATH=${HOME_DIR}/logs

# Install runtime dependencies
# These packages are required for the application to run in production
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -u ${CONTAINER_APP_UID} -s /bin/bash -d ${HOME_DIR} ${CONTAINER_APP_USERNAME} \
    && mkdir -p ${HOME_DIR} ${HOME_DIR}/logs \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR} \
    && mkdir -p /run/sshd \
    && chmod 755 /run/sshd

# Copy everything from builder
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    /home/container_app_user ${HOME_DIR}

# Copy uv's Python installation (venv symlinks point to /root/.local/share/uv/python)
# Make parent directories traversable and uv directory fully accessible
COPY --from=builder /root/.local/share/uv /root/.local/share/uv
RUN chmod 755 /root /root/.local /root/.local/share && chmod -R 755 /root/.local/share/uv

# Fix venv permissions (COPY --chown can break symlink permissions)
RUN chmod -R +x ${PYTHON_ENV_DIR}/bin

USER ${CONTAINER_APP_USERNAME}
WORKDIR ${HOME_DIR}/tt-metal

# Set up bashrc
RUN echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc

EXPOSE 8000
CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && cd ${TT_METAL_HOME}/server/ && source ./run_uvicorn.sh"]
