# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# Local-source variant of vllm.tt-metal.src.dev.Dockerfile.
#
# The stock Dockerfile git-clones tt-metal and vllm-tt-plugin at pinned commits.
# This one COPYs git-archive exports of the local (unpushed) checkouts instead; the
# build context is assembled by scripts/build_local_dsv41_image.sh:
#
#   <ctx>/tt-metal            git archive of tt-metal HEAD + its 3 submodules (no .git), WITHOUT models/
#   <ctx>/tt-metal-models     the models/ tree of that export (python only, copied after the C++ build
#                             so a models/ change does not invalidate the build layers)
#   <ctx>/vllm-tt-plugin      git archive of the vllm-tt-plugin worktree HEAD
#   <ctx>/vllm-tt-metal/...   this repo's vllm-tt-metal/{src,requirements.txt}
#   <ctx>/utils, VERSION, reference_config, workflows, workflow_module, report_module, model_spec.json (generated with MODEL_SPECS_ENV=dev)
#
# Layer caching: the builder stage must not see the commit shas (no ARG/ENV of them before
# the C++ build: an ARG is an implicit env of every later RUN), so a tt-metal commit that
# only touches models/ (or only tt-inference-server / the plugin) reuses build_metal.sh and
# create_venv.sh from the cache. The shas are recorded in BUILD_INFO (runtime stage).
#
# Runtime layout is identical to the stock image (TT_METAL_HOME, vllm_tt_plugin_dir,
# app dir with src/ and utils/, same entrypoint). No weights or caches are baked in.
ARG TT_METAL_DOCKERFILE_URL

# ==============================================================================
# BUILDER STAGE
# ==============================================================================
FROM ${TT_METAL_DOCKERFILE_URL} AS builder

ARG TT_SMI_COMMIT_SHA_OR_TAG=v3.1.1
# cap on compile parallelism (shared build hosts): cmake/ninja and make
ARG BUILD_JOBS=16
ARG CONTAINER_APP_UID=1000
ARG DEBIAN_FRONTEND=noninteractive
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}

ENV SHELL=/bin/bash \
    TZ=America/Los_Angeles \
    CONTAINER_APP_USERNAME=${CONTAINER_APP_USERNAME} \
    ARCH_NAME=blackhole \
    TT_METAL_HOME=${HOME_DIR}/tt-metal \
    CONFIG=Release \
    TT_METAL_ENV=dev \
    VLLM_TARGET_DEVICE="tt" \
    vllm_tt_plugin_dir=${HOME_DIR}/vllm-tt-plugin \
    TT_SMI_DIR=${HOME_DIR}/tt-smi \
    LOGURU_LEVEL=INFO \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo
ENV PYTHONPATH=${TT_METAL_HOME} \
    PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env \
    LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib \
    PATH="$CARGO_HOME/bin:$PATH"

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
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -u ${CONTAINER_APP_UID} -s /bin/bash -d ${HOME_DIR} ${CONTAINER_APP_USERNAME} \
    && mkdir -p ${HOME_DIR} \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR}

RUN if [ -z "${RUSTUP_HOME}" ] || [ -z "${CARGO_HOME}" ]; then echo "RUSTUP_HOME and CARGO_HOME must be set" >&2; exit 1; fi && \
    mkdir -p "${RUSTUP_HOME}" "${CARGO_HOME}" && \
    chown -R ${CONTAINER_APP_UID}:${CONTAINER_APP_UID} "${RUSTUP_HOME}" "${CARGO_HOME}" && \
    chmod -R 775 "${RUSTUP_HOME}" "${CARGO_HOME}"

RUN /bin/bash -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path \
    && . ${CARGO_HOME}/env \
    && rustup update"

ENV UV_HTTP_RETRIES=10 \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    MAKEFLAGS=-j${BUILD_JOBS}

# tt-metal from the local export. The export has no .git, but setuptools_scm and
# create_venv.sh expect a repo: a throwaway one exists only for this RUN and is removed.
COPY tt-metal ${TT_METAL_HOME}
RUN /bin/bash -c "cd ${TT_METAL_HOME} \
    && git init -q && git add -A \
    && git -c user.name=build -c user.email=build@localhost commit -q -m 'local export' \
    && export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0+local \
    && bash ./build_metal.sh \
    && ( for i in 1 2 3 4 5; do CXX=clang++-17 CC=clang-17 bash ./create_venv.sh && exit 0; echo 'create_venv.sh failed, retrying in 30s'; sleep 30; done; exit 1 ) \
    && rm -rf ${TT_METAL_HOME}/.git \
    && { uv cache clean || echo 'WARN: uv cache clean failed'; true; }"

# vllm-tt-plugin from the local export; its own docs/install-vllm-tt.sh owns the vLLM
# pin and dependency overrides (needs PyPI / GitHub raw / download.pytorch.org).
COPY vllm-tt-plugin ${vllm_tt_plugin_dir}
RUN /bin/bash -c "cd ${vllm_tt_plugin_dir} \
    && source ${PYTHON_ENV_DIR}/bin/activate \
    && uv pip install --upgrade pip \
    && source docs/install-vllm-tt.sh \
    && { uv cache clean || echo 'WARN: uv cache clean failed'; true; }"

# models/ (python only) after the C++ build and the venv
COPY tt-metal-models ${TT_METAL_HOME}/models
RUN /bin/bash -c "cd ${TT_METAL_HOME} \
    && source ${PYTHON_ENV_DIR}/bin/activate \
    && if [ -f 'models/demos/qwen25_vl/requirements.txt' ]; then uv pip install -r models/demos/qwen25_vl/requirements.txt; fi \
    && { uv cache clean || echo 'WARN: uv cache clean failed'; true; }"

# tt-smi in its own venv (same pinned public tag as the stock image)
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
# RUNTIME STAGE
# ==============================================================================
FROM ${TT_METAL_DOCKERFILE_URL} AS runtime

LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server \
      org.opencontainers.image.description="DeepSeek-V4.1-Flash on Blackhole Galaxy, built from local (unpushed) tt-metal / tt-inference-server / vllm-tt-plugin sources"

ARG CONTAINER_APP_UID=15863
ARG DEBIAN_FRONTEND=noninteractive
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}
ARG APP_DIR="${HOME_DIR}/app"

ENV SHELL=/bin/bash \
    TZ=America/Los_Angeles \
    CONTAINER_APP_USERNAME=${CONTAINER_APP_USERNAME} \
    ARCH_NAME=blackhole \
    TT_METAL_HOME=${HOME_DIR}/tt-metal \
    CONFIG=Release \
    TT_METAL_ENV=dev \
    VLLM_TARGET_DEVICE="tt" \
    vllm_tt_plugin_dir=${HOME_DIR}/vllm-tt-plugin \
    TT_SMI_DIR=${HOME_DIR}/tt-smi \
    LOGURU_LEVEL=INFO \
    TT_METAL_LOGS_PATH=${HOME_DIR}/logs
ENV PYTHONPATH=${TT_METAL_HOME}:${APP_DIR} \
    PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env \
    LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib

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

COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    ${TT_METAL_HOME} ${TT_METAL_HOME}
# editable-install target of the plugin: must keep the builder's absolute path
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    ${vllm_tt_plugin_dir} ${vllm_tt_plugin_dir}
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    ${TT_SMI_DIR} ${TT_SMI_DIR}

COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "vllm-tt-metal/src" "${APP_DIR}/src"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "vllm-tt-metal/requirements.txt" "${APP_DIR}/requirements.txt"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "utils" "${APP_DIR}/utils"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "VERSION" "${APP_DIR}/VERSION"
# eval / model-spec configuration (reference_config/evals/eval_config.py, workflows/model_specs)
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "reference_config" "${APP_DIR}/reference_config"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "workflows" "${APP_DIR}/workflows"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "workflow_module" "${APP_DIR}/workflow_module"
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    "report_module" "${APP_DIR}/report_module"

# the whole client side of tt-inference-server (run.py, run_workflows.py, llm_module, test_module, launchers, requirements, ...) so that the
# evals / benchmarks workflows can run from this image against a server; its .git marker (git init in the build script) makes get_repo_root_path work.
# World-writable so a non-root `docker run --user` can create .workflow_venvs / workflow_logs (or bind-mount host dirs there).
COPY ttis ${HOME_DIR}/tt-inference-server
RUN chmod -R a+rwX ${HOME_DIR}/tt-inference-server

# Fix venv symlinks after copy and install additional app requirements
RUN cd ${PYTHON_ENV_DIR}/bin \
    && rm -f python python3 \
    && ln -s /usr/bin/python3 python3 \
    && ln -s python3 python \
    && /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && uv pip install --no-cache-dir -r ${APP_DIR}/requirements.txt \
    && uv cache clean" \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${PYTHON_ENV_DIR}

RUN chmod -R +x ${PYTHON_ENV_DIR}/bin

USER ${CONTAINER_APP_USERNAME}

# Defaults equal to the catalog entry (workflows/model_specs/dev/llm.yaml,
# DeepSeek-V4.1-Flash / BLACKHOLE_GALAXY env_vars). The server also applies the entry
# from model_spec.json; these only matter for direct `--entrypoint python` use.
# DSV41_CKPT and the three cache dirs are HOST paths: bind-mount them at the same path.
ENV TT_METAL_LOGS_PATH=/home/container_app_user/logs \
    CACHE_ROOT=/home/container_app_user/cache_root \
    MODEL_SPECS_JSON_PATH=/home/container_app_user/model_specs/model_spec.json \
    VLLM_TARGET_DEVICE=tt \
    HF_MODEL=deepseek-ai/DeepSeek-V4.1-Flash \
    MESH_DEVICE="(4, 8)" \
    DSV41_LAYERS=0-39 \
    DSV41_ENGRAM_RAM=1 \
    DSV41_POOL_DTYPE=fp8 \
    DSV41_TRACE_REGION=1600000000 \
    DSV41_CKPT=/mnt/tt-data/ssinghal/deepseek-v41-flash \
    DSV41_WEIGHT_CACHE=/home/ttuser/ssinghal/tt-metal-cache/weights \
    DSV41_UNIFIED_CACHE=/home/ttuser/ssinghal/tt-metal-cache/unified \
    TT_METAL_CACHE=/home/ttuser/ssinghal/tt-metal-cache/kernels \
    EXTRA_MODELS_DIR=../../tt-metal/models/demos/blackhole

RUN mkdir -p ${CACHE_ROOT} /home/container_app_user/model_specs
COPY --chown=container_app_user:container_app_user \
    model_spec.json ${MODEL_SPECS_JSON_PATH}

# shas last: an ARG is an implicit env of every later RUN
ARG TT_METAL_COMMIT_SHA_OR_TAG=unknown
ARG TT_INFERENCE_SERVER_COMMIT=unknown
ARG VLLM_TT_PLUGIN_COMMIT=unknown
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG}
RUN printf 'tt-metal=%s\ntt-inference-server=%s\nvllm-tt-plugin=%s\nbuilt=%s\n' \
    "${TT_METAL_COMMIT_SHA_OR_TAG}" "${TT_INFERENCE_SERVER_COMMIT}" "${VLLM_TT_PLUGIN_COMMIT}" "$(date -u +%FT%TZ)" \
    > /home/container_app_user/BUILD_INFO

WORKDIR "${APP_DIR}/src"

# Usage: docker run <image> --model deepseek-ai/DeepSeek-V4.1-Flash --tt-device <device>
ENTRYPOINT ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && exec python run_vllm_api_server.py \"$@\"", "--"]
