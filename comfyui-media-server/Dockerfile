# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# ==============================================================================
# comfyui-media-server image
# ==============================================================================
# Builds a pinned tt-metal from source, then drops this server on top. Trimmed
# from tt-media-server/Dockerfile: no cpp_server, no vLLM plugin, no whisper
# overrides — this server is a Python FastAPI diffusion server only.
#
# NOTE: building tt-metal from source is LONG (~30-60+ min) and heavy. For fast
# local iteration against an already-built host tt-metal, skip Docker entirely
# and use launch_server.sh directly — see LOCAL_TESTING.md (Path 1).
#
# Build:
#   docker build -f comfyui-media-server/Dockerfile \
#     --build-arg TT_METAL_COMMIT_SHA_OR_TAG=<sha> \
#     -t comfyui-media-server:dev comfyui-media-server
# ==============================================================================
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest

LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

ARG DEBIAN_FRONTEND=noninteractive
# The amalgamation branch (samt/standalone-media-20260703) HEAD, which carries the
# tt-metal library changes the runners import. Override with --build-arg.
ARG TT_METAL_COMMIT_SHA_OR_TAG=ce05994325a010962c53b89634feeddf2171a75b
ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG}

ENV SHELL=/bin/bash
ENV CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}

# tt-metal build/runtime vars
ENV TT_METAL_HOME=${HOME_DIR}/tt-metal
ENV CONFIG=Release
ENV TT_METAL_ENV=dev
ENV LOGURU_LEVEL=INFO
ENV PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env
ENV LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib
# where this server's code will live inside the image
ENV SERVER_DIR=${HOME_DIR}/comfyui-media-server
# tt-metal (for ttnn/models.*) AND the server dir (for server/device_specs/utils)
ENV PYTHONPATH=${TT_METAL_HOME}:${SERVER_DIR}
ENV TT_MM_THROTTLE_PERF=5

# Refresh Kitware keyring, then system deps (libgl1 for PIL, ffmpeg for video,
# git-lfs for weights, gosu/misc for parity with tt-media-server).
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update && apt-get install -y \
        gosu libgl1 libsndfile1 ffmpeg wget git git-lfs jq curl unzip build-essential \
        python3-dev libffi-dev libssl-dev protobuf-compiler libprotobuf-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# CMake 3.26.4 (CMP0169 compatibility), matches tt-media-server.
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh -o cmake.sh \
    && chmod +x cmake.sh && ./cmake.sh --skip-license --prefix=/usr/local && rm cmake.sh

# Build tt-metal at the pinned SHA.
RUN git clone https://github.com/tenstorrent/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git fetch origin ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && git submodule update --init --recursive \
    && ./install_dependencies.sh --docker \
    && CXX=clang++-17 CC=clang-17 bash ./create_venv.sh --bundle-python \
    && echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc \
    && bash ./build_metal.sh --release -e \
    && find ${TT_METAL_HOME} -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true

# tt-smi (launch_server.sh uses it for device detection).
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && export CARGO_HOME=${HOME_DIR}/.cargo RUSTUP_HOME=${HOME_DIR}/.rustup \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path \
    && . ${HOME_DIR}/.cargo/env \
    && uv pip install git+https://github.com/tenstorrent/tt-smi \
    && rm -rf ${HOME_DIR}/.rustup ${HOME_DIR}/.cargo"

# Drop this server in and install its extra deps (fastapi + uvicorn) into the venv.
# torch/numpy/PIL/diffusers come from tt-metal's python_env (used by the demos the
# runners import); if a model dep is missing at runtime, add it here.
COPY . ${SERVER_DIR}
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && uv pip install --no-cache-dir -r ${SERVER_DIR}/requirements-server.txt" \
    && chmod +x ${SERVER_DIR}/launch_server.sh ${SERVER_DIR}/stop_server.sh ${SERVER_DIR}/docker-entrypoint.sh

# HF cache defaults; run.py bind-mounts weights/cache at runtime.
ENV HF_HOME=${HOME_DIR}/.cache/huggingface

WORKDIR ${SERVER_DIR}
EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
