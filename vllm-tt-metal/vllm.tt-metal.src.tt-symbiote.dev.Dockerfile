# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# tt_symbiote build variant of vllm.tt-metal.src.dev.Dockerfile.
#
# This image is dedicated to tt_symbiote-impl models. It is identical to the
# tt_transformers (default) image EXCEPT that it ALWAYS installs the tt_symbiote
# library (HuggingFace-shaped TTNN models) into the tt-metal venv. The default
# dev Dockerfile is tt_transformers-only and must NOT install tt_symbiote.
#
# Selection between the two Dockerfiles is driven by build_docker_images.py based
# on whether a tt_symbiote model maps to the (tt_metal, vllm) commit pair.

# Optimized multi-stage build for significantly smaller runtime images
ARG TT_METAL_DOCKERFILE_URL

# ==============================================================================
# BUILDER STAGE - Contains all build dependencies and artifacts
# ==============================================================================
FROM ${TT_METAL_DOCKERFILE_URL} AS builder

# Build arguments
ARG TT_METAL_COMMIT_SHA_OR_TAG
ARG TT_VLLM_COMMIT_SHA_OR_TAG
ARG TT_SMI_COMMIT_SHA_OR_TAG=v3.1.1
ARG CONTAINER_APP_UID=1000
ARG DEBIAN_FRONTEND=noninteractive
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}
# Clang version used to build the tt-metal Python venv. Must match a clang
# toolchain present in the tt-metal base image (TT_METAL_DOCKERFILE_URL), which
# varies by tt-metal commit: pre-bake bases (e.g. c09f09c3) ship clang-20, while
# newer bake bases ship clang-17. Default 20 for the current dots.ocr pin;
# override with --build-arg TT_METAL_CLANG_VERSION=<n> for other commits.
ARG TT_METAL_CLANG_VERSION=20

# Environment variables for build
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
    SHELL=/bin/bash \
    TZ=America/Los_Angeles \
    CONTAINER_APP_USERNAME=${CONTAINER_APP_USERNAME} \
    ARCH_NAME=wormhole_b0 \
    TT_METAL_HOME=${HOME_DIR}/tt-metal \
    CONFIG=Release \
    TT_METAL_ENV=dev \
    VLLM_TARGET_DEVICE="tt" \
    vllm_dir=${HOME_DIR}/vllm \
    TT_SMI_DIR=${HOME_DIR}/tt-smi \
    LOGURU_LEVEL=INFO \
    # Rust build dependencies, for backward compatibility with tt-metal 
    # versions where build Docker image does not have these defined
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo
# Environment variables defined by other env vars
ENV PYTHONPATH=${TT_METAL_HOME} \
    PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env \
    LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib \
    PATH="$CARGO_HOME/bin:$PATH"

# Install only essential build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    python3-dev \
    git \
    build-essential \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libsndfile1 \
    libffi-dev \
    libssl-dev \
    # pyluwen build dependencies (Rust package with protobuf)
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# User setup
RUN useradd -u ${CONTAINER_APP_UID} -s /bin/bash -d ${HOME_DIR} ${CONTAINER_APP_USERNAME} \
    && mkdir -p ${HOME_DIR} \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR}

# Give user write access to Rust directories (fail if env vars are missing)
RUN if [ -z "${RUSTUP_HOME}" ] || [ -z "${CARGO_HOME}" ]; then echo "RUSTUP_HOME and CARGO_HOME must be set" >&2; exit 1; fi && \
    mkdir -p "${RUSTUP_HOME}" "${CARGO_HOME}" && \
    chown -R ${CONTAINER_APP_UID}:${CONTAINER_APP_UID} "${RUSTUP_HOME}" "${CARGO_HOME}" && \
    chmod -R 775 "${RUSTUP_HOME}" "${CARGO_HOME}"

RUN /bin/bash -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path \
    && . ${CARGO_HOME}/env \
    && rustup update"

# Build tt-metal - clone with minimal history, build, and clean
RUN /bin/bash -c "git clone https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && git submodule update --init --recursive \
    && bash ./build_metal.sh \
    && CXX=clang++-${TT_METAL_CLANG_VERSION} CC=clang-${TT_METAL_CLANG_VERSION} bash ./create_venv.sh \
    && source ${PYTHON_ENV_DIR}/bin/activate \
    && if [ -f 'models/demos/qwen25_vl/requirements.txt' ]; then uv pip install -r models/demos/qwen25_vl/requirements.txt; fi \
    && rm -rf ${TT_METAL_HOME}/.git"

# Build vllm - clone with minimal history and clean
# Use uv pip to match tt-metal's package manager (see tt-metal commit 29d59d1)
# Use --index-strategy unsafe-best-match to allow uv to find packages across all indexes
RUN /bin/bash -c "git clone https://github.com/tenstorrent/vllm.git ${vllm_dir} \
    && cd ${vllm_dir} \
    && git checkout ${TT_VLLM_COMMIT_SHA_OR_TAG} \
    && source ${PYTHON_ENV_DIR}/bin/activate \
    && uv pip install --upgrade pip \
    && VLLM_TARGET_DEVICE=empty uv pip install --index-strategy unsafe-best-match -e . --extra-index-url https://download.pytorch.org/whl/cpu \
    && if [ -d plugins/vllm-tt-plugin ]; then uv pip install --index-strategy unsafe-best-match -e 'plugins/vllm-tt-plugin[runtime]' --extra-index-url https://download.pytorch.org/whl/cpu; fi \
    && rm -rf ${vllm_dir}/.git"

# Build tt-smi in separate venv to avoid conflicts with tt-metal venv
RUN /bin/bash -c "git clone https://github.com/tenstorrent/tt-smi.git ${TT_SMI_DIR} \
    && cd ${TT_SMI_DIR} \
    && git checkout ${TT_SMI_COMMIT_SHA_OR_TAG} \
    && python3 -m venv .venv \
    && source .venv/bin/activate \
    && pip3 install --upgrade pip \
    && source ${CARGO_HOME}/env \
    && pip3 install . \
    && rm -rf ${TT_SMI_DIR}/.git"

# ==============================================================================
# RUNTIME STAGE - Minimal dependencies for running the application
# ==============================================================================
FROM ${TT_METAL_DOCKERFILE_URL} AS runtime

LABEL maintainer="Tom Stesco <tstesco@tenstorrent.com>" \
    org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

# IDENTICAL arguments and environment as builder stage
ARG TT_METAL_COMMIT_SHA_OR_TAG
ARG CONTAINER_APP_UID=15863
ARG DEBIAN_FRONTEND=noninteractive
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}
ARG APP_DIR="${HOME_DIR}/app"
# tt_symbiote release to install. As of tt_symbiote 0.1.5, ttnn is NOT a package
# dependency, so a normal install can never overwrite the source-built ttnn (from
# TT_METAL_COMMIT_SHA_OR_TAG). REQUIRED for this tt_symbiote image (the build
# fails below if unset).
ARG TT_SYMBIOTE_VERSION=""
# Extra package index tt_symbiote is pulled from. Defaults to the production PyPI
# index (tt_symbiote 0.1.5+ is published there). To install a pre-release from
# TestPyPI for validation, override the build-arg:
#     --build-arg TT_SYMBIOTE_INDEX_URL=https://test.pypi.org/simple/
# Used as an --extra-index-url so tt_symbiote can come from TestPyPI while its
# real deps (torch / transformers) still resolve from the default PyPI. ttnn is
# never pulled from any index (source-build-only).
ARG TT_SYMBIOTE_INDEX_URL="https://pypi.org/simple/"

# IDENTICAL environment variables as builder stage
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
    SHELL=/bin/bash \
    TZ=America/Los_Angeles \
    CONTAINER_APP_USERNAME=${CONTAINER_APP_USERNAME} \
    ARCH_NAME=wormhole_b0 \
    TT_METAL_HOME=${HOME_DIR}/tt-metal \
    CONFIG=Release \
    TT_METAL_ENV=dev \
    VLLM_TARGET_DEVICE="tt" \
    vllm_dir=${HOME_DIR}/vllm \
    TT_SMI_DIR=${HOME_DIR}/tt-smi \
    LOGURU_LEVEL=INFO \
    TT_METAL_LOGS_PATH=${HOME_DIR}/logs
# Environment variables defined by other env vars.
# ${APP_DIR}/src is on PYTHONPATH so tt-inference-server serving adapters (e.g.
# tt_symbiote_generators) are importable by vLLM worker subprocesses when the
# ModelRegistry lazy-import string is resolved.
ENV PYTHONPATH=${TT_METAL_HOME}:${APP_DIR}:${APP_DIR}/src \
    PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env \
    LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib

# Install only runtime dependencies + create IDENTICAL user
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    libgl1 \
    libsndfile1 \
    ca-certificates \
    wget \
    nano \
    acl \
    jq \
    vim \
    # user convenience deps
    htop \
    screen \
    tmux \
    unzip \
    zip \
    curl \
    iputils-ping \
    rsync \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && useradd -u ${CONTAINER_APP_UID} -s /bin/bash -d ${HOME_DIR} ${CONTAINER_APP_USERNAME} \
    && mkdir -p ${HOME_DIR} ${APP_DIR} ${HOME_DIR}/logs \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR} \
    && echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc

# Copy complete tt-metal installation including virtual environment
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    ${TT_METAL_HOME} ${TT_METAL_HOME}

# Copy complete vllm installation  
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    ${vllm_dir} ${vllm_dir}

# Copy complete tt-smi installation  
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    ${TT_SMI_DIR} ${TT_SMI_DIR}

# Copy application files
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "vllm-tt-metal/src" "${APP_DIR}/src"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "vllm-tt-metal/requirements.txt" "${APP_DIR}/requirements.txt"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "utils" "${APP_DIR}/utils"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "VERSION" "${APP_DIR}/VERSION"

# Fix venv symlinks after copy and install additional app requirements
RUN cd ${PYTHON_ENV_DIR}/bin \
    && rm -f python python3 \
    && ln -s /usr/bin/python3 python3 \
    && ln -s python3 python \
    && /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && uv pip install --no-cache-dir -r ${APP_DIR}/requirements.txt \
    && uv cache clean" \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${PYTHON_ENV_DIR}

# Install tt_symbiote (HuggingFace-shaped TTNN model library) with its real
# dependencies. As of tt_symbiote 0.1.5, ttnn is NOT a package dependency
# (source-build-only), so a normal install pulls only torch / transformers 5.x /
# etc. and can NEVER overwrite the ttnn already built from source at
# TT_METAL_COMMIT_SHA_OR_TAG. (The previous --no-deps + "install deps except ttnn"
# dance was only needed for <=0.1.4, which still pinned ttnn==0.68.0.)
#
# This is the tt_symbiote-only image, so TT_SYMBIOTE_VERSION is REQUIRED and the
# build fails fast if it is unset. By default tt_symbiote and all of its deps
# resolve from the production PyPI (TT_SYMBIOTE_INDEX_URL=https://pypi.org/simple/),
# i.e. a plain `pip install tt_symbiote==<version>`. The --extra-index-url +
# --index-strategy unsafe-best-match flags are no-ops against that default; they
# only matter when TT_SYMBIOTE_INDEX_URL is overridden to a pre-release index
# (e.g. TestPyPI), where they let tt_symbiote come from the override index while
# its pinned deps (e.g. transformers==5.9.0) still resolve from PyPI.
RUN if [ -z "${TT_SYMBIOTE_VERSION}" ]; then \
    echo "ERROR: TT_SYMBIOTE_VERSION build-arg is required for the tt_symbiote image" >&2; \
    exit 1; \
    fi \
    && /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && uv pip install --no-cache-dir --index-strategy unsafe-best-match --extra-index-url \"${TT_SYMBIOTE_INDEX_URL}\" tt_symbiote==${TT_SYMBIOTE_VERSION} \
    && uv cache clean" \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${PYTHON_ENV_DIR}

# Cap the FastAPI/Starlette web stack to a version compatible with vLLM's
# prometheus-fastapi-instrumentator. vLLM only sets a LOWER bound
# (fastapi[standard]>=0.115.0), so a fresh build floats up to FastAPI 0.137 /
# Starlette 1.x, whose `_IncludedRouter` route objects break the instrumentator
# (AttributeError: '_IncludedRouter' object has no attribute 'path'), which 500s
# every API request — including /health. Capping to the 0.115 line pulls
# Starlette 0.46.x, which the instrumentator handles. NOTE: this is vLLM web-stack
# drift, not tt_symbiote-specific; the tt_transformers (default) image has the
# same latent issue and likely needs the same cap (tracked as follow-up).
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && uv pip install --no-cache-dir 'fastapi>=0.115,<0.116' \
    && uv cache clean" \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${PYTHON_ENV_DIR}

# Fix venv permissions (COPY --chown can break symlink permissions)
RUN chmod -R +x ${PYTHON_ENV_DIR}/bin

# Switch to non-root user for runtime
USER ${CONTAINER_APP_USERNAME}

# Environment variable defaults (can be overridden at runtime with -e)
ENV TT_METAL_LOGS_PATH=/home/container_app_user/logs \
    CACHE_ROOT=/home/container_app_user/cache_root \
    MODEL_SPECS_JSON_PATH=/home/container_app_user/model_specs/model_spec.json \
    VLLM_TARGET_DEVICE=tt \
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

# Create cache_root directory as non-root user to seed Docker volume with correct ownership
RUN mkdir -p ${CACHE_ROOT}

# Copy pre-generated model specs JSON
RUN mkdir -p /home/container_app_user/model_specs
COPY --chown=container_app_user:container_app_user \
    model_spec.json ${MODEL_SPECS_JSON_PATH}

# Set working directory and entrypoint
WORKDIR "${APP_DIR}/src"

# Usage: docker run <image> --model <hf_repo> --device <device_type>
ENTRYPOINT ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && exec python run_vllm_api_server.py \"$@\"", "--"]
