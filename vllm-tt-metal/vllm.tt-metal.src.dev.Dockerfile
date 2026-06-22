# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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
ARG INSTALL_GEMMA4_REQUIREMENTS=0
ARG CONTAINER_APP_UID=1000
ARG DEBIAN_FRONTEND=noninteractive
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}

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
    && CXX=clang++-17 CC=clang-17 bash ./create_venv.sh \
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

# Optional Gemma 4 deps (transformers 5.x). Installed after vLLM so they override 4.x pins.
ARG INSTALL_GEMMA4_REQUIREMENTS=0
RUN /bin/bash -c "if [ \"${INSTALL_GEMMA4_REQUIREMENTS}\" = \"1\" ]; then \
    source ${PYTHON_ENV_DIR}/bin/activate \
    && gemma4_reqs='${TT_METAL_HOME}/models/demos/gemma4/requirements.txt' \
    && if [ ! -f \"\${gemma4_reqs}\" ]; then echo \"INSTALL_GEMMA4_REQUIREMENTS=1 but \${gemma4_reqs} not found\" >&2; exit 1; fi \
    && uv pip install -r \"\${gemma4_reqs}\"; \
    fi"

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
# Environment variables defined by other env vars
ENV PYTHONPATH=${TT_METAL_HOME}:${APP_DIR} \
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

# Bake Gemma 4 vLLM-side patches into the image so it runs correctly WITHOUT
# --dev-mode (which otherwise bind-mounts these from the host repo, see
# workflows/run_docker_server.py::_vllm_tt_metal_dev_mounts). Gated on
# INSTALL_GEMMA4_REQUIREMENTS so non-Gemma images built from this Dockerfile are
# untouched. These three patches are pinned to the vLLM commit (3334377): the
# unified platform registration plus the gemma4 reasoning parser.
#
# NOTE: As of the tt-metal main bump (3f727969e22) we NO LONGER override
# models/demos/gemma4/tt/generator_vllm.py. Our old gemma4_generator_vllm_9970093.py
# patch was pinned to the 9970093 API (and force-disabled prefill tracing); main
# ships a native vLLM bridge (generator_vllm.py + generator_trace.py) that we now
# use as-is. The legacy patch file is kept in vllm-tt-metal/patches/ for reference
# only and is intentionally not applied.
ARG INSTALL_GEMMA4_REQUIREMENTS=0
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "vllm-tt-metal/patches" "${APP_DIR}/patches"
RUN /bin/bash -c "if [ \"${INSTALL_GEMMA4_REQUIREMENTS}\" = \"1\" ]; then \
    set -e; \
    p='${APP_DIR}/patches'; \
    plat='${vllm_dir}/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py'; \
    rparser='${vllm_dir}/vllm/reasoning/gemma4_reasoning_parser.py'; \
    rinit='${vllm_dir}/vllm/reasoning/__init__.py'; \
    for f in gemma4_platform_vllm_3334377.py gemma4_reasoning_parser_vllm_3334377.py gemma4_reasoning_init_vllm_3334377.py; do \
        if [ ! -f \"\${p}/\${f}\" ]; then echo \"Missing Gemma4 patch: \${p}/\${f}\" >&2; exit 1; fi; \
    done; \
    cp \"\${p}/gemma4_platform_vllm_3334377.py\" \"\${plat}\"; \
    cp \"\${p}/gemma4_reasoning_parser_vllm_3334377.py\" \"\${rparser}\"; \
    cp \"\${p}/gemma4_reasoning_init_vllm_3334377.py\" \"\${rinit}\"; \
    chown ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \"\${plat}\" \"\${rparser}\" \"\${rinit}\"; \
    echo '✅ Baked Gemma4 vLLM-side patches into image (native tt-metal generator_vllm.py)'; \
    else echo 'Skipping Gemma4 patch bake (INSTALL_GEMMA4_REQUIREMENTS != 1)'; fi"

# Fix venv symlinks after copy and install additional app requirements
RUN cd ${PYTHON_ENV_DIR}/bin \
    && rm -f python python3 \
    && ln -s /usr/bin/python3 python3 \
    && ln -s python3 python \
    && /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && uv pip install --no-cache-dir -r ${APP_DIR}/requirements.txt \
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
