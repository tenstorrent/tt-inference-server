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
ARG TT_QUETZAL_COMMIT_SHA=""
ARG TT_METAL_PATCHSET_SHA256=""
ARG TT_METAL_PATCHSET_MANIFEST_SHA256=""
ARG TT_SMI_COMMIT_SHA_OR_TAG=v3.1.1
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
    vllm_tt_plugin_dir=${HOME_DIR}/vllm-tt-plugin \
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

# BuildKit receives this named context from an Actions checkout exported with
# git archive. It contains no .git directory or credentials. Native builds pass
# an empty context and leave TT_QUETZAL_COMMIT_SHA unset.
COPY --from=quetzal_src / /tmp/quetzal-source/
COPY scripts/validate_quetzal_serve_environment.py \
    /tmp/validate_quetzal_serve_environment.py
COPY tt-vllm-plugin/pyproject.toml \
    /tmp/ttis-vllm-plugin-pyproject.toml

# Give user write access to Rust directories (fail if env vars are missing)
RUN if [ -z "${RUSTUP_HOME}" ] || [ -z "${CARGO_HOME}" ]; then echo "RUSTUP_HOME and CARGO_HOME must be set" >&2; exit 1; fi && \
    mkdir -p "${RUSTUP_HOME}" "${CARGO_HOME}" && \
    chown -R ${CONTAINER_APP_UID}:${CONTAINER_APP_UID} "${RUSTUP_HOME}" "${CARGO_HOME}" && \
    chmod -R 775 "${RUSTUP_HOME}" "${CARGO_HOME}"

RUN /bin/bash -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path \
    && . ${CARGO_HOME}/env \
    && rustup update"

# download.pytorch.org intermittently returns 503 during dependency installs;
# raise uv's HTTP retry count (default 3) for every uv invocation in create_venv.sh.
ENV UV_HTTP_RETRIES=10

# Build tt-metal - clone with minimal history, build, and clean
# uv's cache grows to ~3G here and ~11G after the vLLM install; the builder stage is
# discarded, but its layers still fill the container storage fs during image export.
# A full-history clone of tt-metal has taken over an hour on CI, connection dropped ("fatal: early
# EOF"). Only the pinned commit is needed, so fetch just that (matches the shallow
# clone already used by tt-media-server/Dockerfile).
RUN set -eux; \
    git clone --depth 1 https://github.com/tenstorrent-metal/tt-metal.git "${TT_METAL_HOME}"; \
    git -C "${TT_METAL_HOME}" fetch --depth 1 origin "${TT_METAL_COMMIT_SHA_OR_TAG}"; \
    git -C "${TT_METAL_HOME}" checkout --detach "${TT_METAL_COMMIT_SHA_OR_TAG}"; \
    if [ -n "${TT_QUETZAL_COMMIT_SHA}" ]; then \
      test "$(cat /tmp/quetzal-source/.tt-quetzal-commit)" = "${TT_QUETZAL_COMMIT_SHA}"; \
      printf '%s' "${TT_METAL_PATCHSET_SHA256}" | grep -Eq '^[0-9a-f]{64}$'; \
      printf '%s' "${TT_METAL_PATCHSET_MANIFEST_SHA256}" | grep -Eq '^[0-9a-f]{64}$'; \
      test "${TT_METAL_PATCHSET_SHA256}" = "${TT_METAL_PATCHSET_MANIFEST_SHA256}"; \
      echo "${TT_METAL_PATCHSET_MANIFEST_SHA256}  /tmp/quetzal-source/patches/tt-metal/gdn-productization-v1.json" | sha256sum --check -; \
      python3 /tmp/quetzal-source/tools/tt_metal_patchset.py \
        --repo "${TT_METAL_HOME}" \
        --manifest /tmp/quetzal-source/patches/tt-metal/gdn-productization-v1.json \
        --apply > /tmp/patchset-apply.json; \
      grep -q '"status": "pass"' /tmp/patchset-apply.json; \
      python3 /tmp/quetzal-source/tools/tt_metal_patchset.py \
        --repo "${TT_METAL_HOME}" \
        --manifest /tmp/quetzal-source/patches/tt-metal/gdn-productization-v1.json \
        > "${TT_METAL_HOME}/.ttq-patchset-admission.json"; \
      grep -q '"status": "pass"' "${TT_METAL_HOME}/.ttq-patchset-admission.json"; \
      printf '%s\n' \
        "{\"base_revision\":\"${TT_METAL_COMMIT_SHA_OR_TAG}\",\"patchset\":\"gdn-productization-v1\",\"patchset_sha256\":\"${TT_METAL_PATCHSET_SHA256}\",\"manifest_sha256\":\"${TT_METAL_PATCHSET_MANIFEST_SHA256}\"}" \
        > "${TT_METAL_HOME}/.ttq-runtime-identity.json"; \
    fi; \
    git -C "${TT_METAL_HOME}" submodule update --init --recursive; \
    cd "${TT_METAL_HOME}"; \
    bash ./build_metal.sh; \
    ( for i in 1 2 3 4 5; do CXX=clang++-17 CC=clang-17 bash ./create_venv.sh && exit 0; echo 'create_venv.sh failed, retrying in 30s'; sleep 30; done; exit 1 ); \
    . "${PYTHON_ENV_DIR}/bin/activate"; \
    if [ -f 'models/demos/qwen25_vl/requirements.txt' ]; then uv pip install -r models/demos/qwen25_vl/requirements.txt; fi; \
    rm -rf "${TT_METAL_HOME}/.git"; \
    { uv cache clean || echo 'WARN: uv cache clean failed'; true; }

# Build vllm-tt-plugin - clone with minimal history and clean.
# The plugin owns the vLLM version pin and its dependency overrides, so the
# install is delegated to its own docs/install-vllm-tt.sh rather than restated here
RUN /bin/bash -c "git clone https://github.com/tenstorrent/vllm-tt-plugin.git ${vllm_tt_plugin_dir} \
    && cd ${vllm_tt_plugin_dir} \
    && git checkout ${TT_VLLM_COMMIT_SHA_OR_TAG} \
    && export VIRTUAL_ENV=${PYTHON_ENV_DIR} \
    && export PATH=${PYTHON_ENV_DIR}/bin:\${PATH} \
    && export UV_PYTHON=${PYTHON_ENV_DIR}/bin/python \
    && uv pip install --python ${PYTHON_ENV_DIR}/bin/python --upgrade pip \
    && source docs/install-vllm-tt.sh \
    && ${PYTHON_ENV_DIR}/bin/python -c \"import importlib.metadata as m; assert m.distribution('vllm-tt-plugin').version\" \
    && rm -rf ${vllm_tt_plugin_dir}/.git \
    && { uv cache clean || echo 'WARN: uv cache clean failed'; true; }"

# Optional generated-Quetzal image hook. A Quetzal-capable image must supply an
# immutable 40-hex commit; ordinary native images leave the argument empty. We
# retain the built wheel in the image build artifacts and install that exact
# wheel without dependencies, so the source/ref and installed payload are both
# auditable and cannot replace the image's pinned tt-metal/vLLM stack. vLLM
# discovers quetzal_model_registry from the wheel's distribution metadata.
RUN set -eu; \
    mkdir -p "${HOME_DIR}/quetzal-runtime/mesh_graph_descriptors" \
             "${HOME_DIR}/quetzal-runtime/wheels"; \
    printf '%s\n' '{"schema":"ttis.quetzal-serve-environment.v1","status":"native-only"}' \
      > "${HOME_DIR}/quetzal-runtime/qualified-environment.json"; \
    if [ -n "${TT_QUETZAL_COMMIT_SHA}" ]; then \
      printf '%s' "${TT_QUETZAL_COMMIT_SHA}" | grep -Eq '^[0-9a-f]{40}$' \
      || { echo 'TT_QUETZAL_COMMIT_SHA must be a lowercase 40-hex commit' >&2; exit 1; }; \
      test "$(cat /tmp/quetzal-source/.tt-quetzal-commit)" = "${TT_QUETZAL_COMMIT_SHA}"; \
      cd /tmp/quetzal-source; \
      export VIRTUAL_ENV="${PYTHON_ENV_DIR}"; \
      export PATH="${PYTHON_ENV_DIR}/bin:${PATH}"; \
      export UV_PYTHON="${PYTHON_ENV_DIR}/bin/python"; \
      "${PYTHON_ENV_DIR}/bin/python" /tmp/validate_quetzal_serve_environment.py \
        --source /tmp/quetzal-source \
        --source-revision "${TT_QUETZAL_COMMIT_SHA}" \
        --plugin-project /tmp/ttis-vllm-plugin-pyproject.toml \
        --check-installed \
        --receipt "${HOME_DIR}/quetzal-runtime/qualified-environment.json"; \
      uv build --wheel --out-dir "${HOME_DIR}/quetzal-runtime/wheels"; \
      test "$(find "${HOME_DIR}/quetzal-runtime/wheels" -maxdepth 1 -type f -name '*.whl' | wc -l)" -eq 1; \
      quetzal_wheel="$(find "${HOME_DIR}/quetzal-runtime/wheels" -maxdepth 1 -type f -name '*.whl')"; \
      uv pip install --python "${PYTHON_ENV_DIR}/bin/python" --no-cache-dir --no-deps "${quetzal_wheel}"; \
      cp serving/mesh_graph_descriptors/p150_x4_2ch_mesh_graph_descriptor.textproto \
         "${HOME_DIR}/quetzal-runtime/mesh_graph_descriptors/"; \
      rm -rf /tmp/quetzal-source; \
      { uv cache clean || echo 'WARN: uv cache clean failed'; true; }; \
    else \
      echo 'Building native-only image (TT_QUETZAL_COMMIT_SHA unset)'; \
    fi

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
ARG TT_QUETZAL_COMMIT_SHA=""
ARG TT_METAL_PATCHSET_SHA256=""
ARG TT_METAL_PATCHSET_MANIFEST_SHA256=""
ARG CONTAINER_APP_UID=15863
ARG DEBIAN_FRONTEND=noninteractive
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}
ARG APP_DIR="${HOME_DIR}/app"

LABEL org.opencontainers.image.quetzal.revision=${TT_QUETZAL_COMMIT_SHA} \
    org.opencontainers.image.tt-metal.patchset.sha256=${TT_METAL_PATCHSET_SHA256} \
    org.opencontainers.image.tt-metal.patchset.manifest.sha256=${TT_METAL_PATCHSET_MANIFEST_SHA256}

# IDENTICAL environment variables as builder stage
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
    TT_QUETZAL_COMMIT_SHA=${TT_QUETZAL_COMMIT_SHA} \
    TT_METAL_PATCHSET_SHA256=${TT_METAL_PATCHSET_SHA256} \
    TT_METAL_PATCHSET_MANIFEST_SHA256=${TT_METAL_PATCHSET_MANIFEST_SHA256} \
    SHELL=/bin/bash \
    TZ=America/Los_Angeles \
    CONTAINER_APP_USERNAME=${CONTAINER_APP_USERNAME} \
    ARCH_NAME=wormhole_b0 \
    TT_METAL_HOME=${HOME_DIR}/tt-metal \
    CONFIG=Release \
    TT_METAL_ENV=dev \
    VLLM_TARGET_DEVICE="tt" \
    vllm_tt_plugin_dir=${HOME_DIR}/vllm-tt-plugin \
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

# Copy the vllm-tt-plugin source tree. This is the editable-install target, so it
# must land at the same absolute path as in the builder or the .pth link breaks.
# vLLM itself needs no COPY of its own: it is a regular (non-editable) install
# inside ${PYTHON_ENV_DIR}/site-packages, already copied with TT_METAL_HOME above.
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    ${vllm_tt_plugin_dir} ${vllm_tt_plugin_dir}

# Stable, catalog-owned locations for the P150x4 descriptor and the exact wheel
# installed in the builder. Both directories remain empty for native-only
# images; Quetzal admission still requires the installed entry point.
RUN mkdir -p /opt/quetzal/mesh_graph_descriptors /opt/quetzal/wheels
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    ${HOME_DIR}/quetzal-runtime/mesh_graph_descriptors/ \
    /opt/quetzal/mesh_graph_descriptors/
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    ${HOME_DIR}/quetzal-runtime/wheels/ \
    /opt/quetzal/wheels/
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    ${HOME_DIR}/quetzal-runtime/qualified-environment.json \
    /opt/quetzal/qualified-environment.json

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
