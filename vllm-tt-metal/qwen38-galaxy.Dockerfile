# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# Context: this repository plus a generated qwen38-bundle/ directory.
# Build arguments and source hashes are recorded in the bundle manifest.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS builder
ARG METAL_SHA
ARG MODEL_SHA
ARG PLUGIN_SHA
ARG BUILD_JOBS=24
ENV TT_METAL_HOME=/home/container_app_user/tt-metal \
    PYTHON_ENV_DIR=/home/container_app_user/tt-metal/python_env \
    VIRTUAL_ENV=/home/container_app_user/tt-metal/python_env \
    VLLM_TARGET_DEVICE=tt \
    UV_HTTP_RETRIES=10 \
    UV_NO_CACHE=1 \
    ARCH_NAME=blackhole
SHELL ["/bin/bash", "-e", "-o", "pipefail", "-c"]
# A fresh exact-commit fetch avoids cloning an unrelated branch or its history.
RUN for pin in "${METAL_SHA}" "${MODEL_SHA}" "${PLUGIN_SHA}"; do \
      [[ "$pin" =~ ^[0-9a-f]{40}$ ]] || exit 2; done \
    && git init "${TT_METAL_HOME}" \
    && git -C "${TT_METAL_HOME}" remote add origin https://github.com/tenstorrent/tt-metal.git \
    && git -C "${TT_METAL_HOME}" fetch --depth 1 origin "${METAL_SHA}" \
    && git -C "${TT_METAL_HOME}" checkout --detach FETCH_HEAD \
    && test "$(git -C "${TT_METAL_HOME}" rev-parse HEAD)" = "${METAL_SHA}" \
    && git -C "${TT_METAL_HOME}" submodule update --init --recursive --jobs 8
WORKDIR ${TT_METAL_HOME}
RUN CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}" ./build_metal.sh --release \
    && ./create_venv.sh --python-version 3.10 --bundle-python
ENV PATH=/home/container_app_user/tt-metal/python_env/bin:${PATH} \
    PYTHONPATH=/home/container_app_user/tt-metal:/home/container_app_user/tt-metal/tools \
    LD_LIBRARY_PATH=/home/container_app_user/tt-metal/build/lib
RUN git init /home/container_app_user/vllm-tt-plugin \
    && git -C /home/container_app_user/vllm-tt-plugin remote add origin https://github.com/tenstorrent/vllm-tt-plugin.git \
    && git -C /home/container_app_user/vllm-tt-plugin fetch --depth 1 origin "${PLUGIN_SHA}" \
    && git -C /home/container_app_user/vllm-tt-plugin checkout --detach FETCH_HEAD \
    && test "$(git -C /home/container_app_user/vllm-tt-plugin rev-parse HEAD)" = "${PLUGIN_SHA}" \
    && cd /home/container_app_user/vllm-tt-plugin \
    && source docs/install-vllm-tt.sh
# Only the model subtree comes from the experiment; the compiled runtime stays
# at METAL_SHA, matching the hardware run. Common sampling stays at METAL_SHA.
RUN git fetch --depth 1 origin "${MODEL_SHA}" \
    && test "$(git rev-parse FETCH_HEAD)" = "${MODEL_SHA}" \
    && git restore --source "${MODEL_SHA}" --worktree -- models/demos/qwen38_27b_qb2
COPY qwen38-bundle /home/container_app_user/qwen38-bundle
COPY vllm-tt-metal/requirements.txt /tmp/ttis-requirements.txt
RUN uv pip install -r /tmp/ttis-requirements.txt \
    && python /home/container_app_user/qwen38-bundle/verify_image.py \
         --metal-sha "${METAL_SHA}" --model-sha "${MODEL_SHA}" --plugin-sha "${PLUGIN_SHA}" \
    && uv pip freeze > /home/container_app_user/qwen38-bundle/python-packages.txt \
    && rm -rf "${TT_METAL_HOME}/.git" /home/container_app_user/vllm-tt-plugin/.git

ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS runtime
ARG TTIS_SHA
ARG METAL_SHA
ARG MODEL_SHA
ARG PLUGIN_SHA
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-inference-server" \
    org.opencontainers.image.revision="${TTIS_SHA}" \
    com.tenstorrent.metal.revision="${METAL_SHA}" \
    com.tenstorrent.qwen.model-source-revision="${MODEL_SHA}" \
    com.tenstorrent.vllm-plugin.revision="${PLUGIN_SHA}" \
    com.tenstorrent.qualification="experimental"
RUN useradd --uid 1000 --create-home --shell /bin/bash container_app_user
COPY --from=builder --chown=1000:1000 /home/container_app_user /home/container_app_user
COPY --chown=1000:1000 vllm-tt-metal/src /home/container_app_user/app/src
COPY --chown=1000:1000 utils /home/container_app_user/app/utils
COPY --chown=1000:1000 VERSION /home/container_app_user/app/VERSION
ENV TT_METAL_HOME=/home/container_app_user/tt-metal \
    PYTHON_ENV_DIR=/home/container_app_user/tt-metal/python_env \
    VIRTUAL_ENV=/home/container_app_user/tt-metal/python_env \
    PATH=/home/container_app_user/tt-metal/python_env/bin:${PATH} \
    PYTHONPATH=/home/container_app_user/tt-metal:/home/container_app_user/tt-metal/tools:/home/container_app_user/app \
    LD_LIBRARY_PATH=/home/container_app_user/tt-metal/build/lib \
    EXTRA_MODELS_DIR=/home/container_app_user/tt-metal/models/demos \
    ARCH_NAME=blackhole \
    VLLM_TARGET_DEVICE=tt \
    RUNTIME_MODEL_SPEC_JSON_PATH=/home/container_app_user/qwen38-bundle/runtime-model-spec.json \
    CACHE_ROOT=/home/container_app_user/cache_root \
    TT_METAL_CACHE=/home/container_app_user/cache_root/jit \
    TT_METAL_LOGS_PATH=/home/container_app_user/cache_root/logs \
    MODEL_WEIGHTS_DIR=/mnt/hf-cache \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=8
USER container_app_user
RUN mkdir -p "${CACHE_ROOT}/jit" "${TT_METAL_LOGS_PATH}" \
    && python /home/container_app_user/qwen38-bundle/verify_image.py
WORKDIR /home/container_app_user/app/src
ENTRYPOINT ["/home/container_app_user/tt-metal/python_env/bin/python", "/home/container_app_user/app/src/run_vllm_api_server.py"]
