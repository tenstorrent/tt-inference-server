# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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
ARG TT_METAL_COMMIT_SHA_OR_TAG=915d9e67ad572b97fbaf5339bd3e518db6dc05e3

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
    libcurl4-openssl-dev \
    iputils-ping \
    rsync \
    && rm -rf /var/lib/apt/lists/* \
    && protoc --version

# Install CMake 3.26.4 to ensure compatibility with CMP0169
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh -o cmake.sh \
    && chmod +x cmake.sh \
    && ./cmake.sh --skip-license --prefix=/usr/local \
    && rm cmake.sh \
    && /usr/local/bin/cmake --version

# download.pytorch.org intermittently returns 503 during dependency installs;
# raise uv's HTTP retry count (default 3) for every uv invocation in create_venv.sh.
ENV UV_HTTP_RETRIES=10
# create_venv.sh installs a uv-managed Python interpreter. The build runs as
# root, so uv's default install dir (~/.local/share/uv) is NOT copied into the
# runtime stage (which only copies ${HOME_DIR}). Newer tt-metal handles this by
# deep-copying the interpreter into the venv (--bundle-python); for older pins
# that predate the flag, keep the interpreter under TT_METAL_HOME so the venv's
# symlinks still resolve after the copy.
ENV UV_PYTHON_INSTALL_DIR=${TT_METAL_HOME}/uv-python

# build tt-metal
#
# Build-only content is reclaimed at the end of THIS RUN rather than in the cleanup
# step at the end of the builder stage. Deleting a file in a later layer only writes a
# whiteout: the bytes stay in this layer's snapshot and occupy /var/lib/containerd for
# the whole build, including the final export and unpack where the build has been
# running out of disk. Removing them here means they are never committed at all.
# The resulting image is unchanged -- this only lowers disk during the build.
RUN git clone --depth 1 https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git fetch --depth 1 origin ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && git checkout ${TT_METAL_COMMIT_SHA_OR_TAG} \
    && git submodule update --init --recursive --depth 1 \
    && ./install_dependencies.sh --docker \
    # Older pinned tt-metal commits (e.g. BGE's 65718bb) predate create_venv.sh's
    # --bundle-python flag and exit 1 on unknown arguments, so only pass it when
    # the script supports it. The test -f guard makes a silent venv failure abort
    # the build here instead of surfacing as a confusing error many layers later.
    && if grep -q -- '--bundle-python' ./create_venv.sh; then \
         CXX=clang++-17 CC=clang-17 bash ./create_venv.sh --bundle-python; \
       else \
         CXX=clang++-17 CC=clang-17 bash ./create_venv.sh; \
       fi \
    && test -f "${PYTHON_ENV_DIR}/bin/activate" \
    && echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc \
    && uv pip install --upgrade setuptools wheel \
    && bash ./build_metal.sh --release -e \
    && bash ./build_metal.sh --release -e --build-metal-tests --configure-only \
    && cmake --build build_Release --target test_system_health -j$(nproc) \
    && uv pip install --upgrade cmake \
    && hash -r \
    && rm -rf ${TT_METAL_HOME}/.cpmcache \
    && rm -rf ${TT_METAL_HOME}/docs \
    && rm -rf ${TT_METAL_HOME}/tests \
    && rm -rf ${TT_METAL_HOME}/tech_reports \
    && rm -rf ${TT_METAL_HOME}/contributing \
    && rm -rf ${TT_METAL_HOME}/releases \
    && rm -rf ${TT_METAL_HOME}/dockerfile \
    && rm -rf ${TT_METAL_HOME}/infra/tests \
    && rm -rf ${TT_METAL_HOME}/tt-train \
    # Parenthesized so `|| true` only forgives the .git cleanup: unscoped, it
    # would swallow a failure anywhere in this && chain and let the build
    # continue without a venv or a tt-metal build.
    && (find ${TT_METAL_HOME} -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true)

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
    && (find ${HOME_DIR} -name '.git' -type d -exec rm -rf {} + 2>/dev/null || true)"

WORKDIR ${HOME_DIR}

# install inference server requirements
ARG APP_DIR="${HOME_DIR}/app"
ENV APP_DIR=${APP_DIR}
WORKDIR ${APP_DIR}
ENV PYTHONPATH=${TT_METAL_HOME}
COPY "tt-media-server" "${TT_METAL_HOME}/server"
COPY "tt-media-server/requirements.txt" "${TT_METAL_HOME}/requirements.txt"
COPY "tt-media-server/audio-requirements.txt" "${TT_METAL_HOME}/audio-requirements.txt"
COPY "tt-media-server/fix_dependencies.sh" "${TT_METAL_HOME}/fix_dependencies.sh"
COPY "tt-media-server/uv-overrides.txt" "${TT_METAL_HOME}/uv-overrides.txt"
# Main venv: Install server requirements (without whisperx/torch pins - those go in audio venv)
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && cd ${TT_METAL_HOME} && uv pip install --no-cache-dir --index-strategy unsafe-best-match -r requirements.txt || \
    (echo 'Requirements installation failed, trying without pyluwen...' && \
    grep -v pyluwen requirements.txt > requirements_no_pyluwen.txt && \
    uv pip install --no-cache-dir --index-strategy unsafe-best-match -r requirements_no_pyluwen.txt) \
    && echo 'Running dependency fixes...' \
    && chmod +x fix_dependencies.sh \
    && ./fix_dependencies.sh"

# Install tt-vllm-plugin
COPY "tt-vllm-plugin" "${TT_METAL_HOME}/tt-vllm-plugin"
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && cd ${TT_METAL_HOME}/tt-vllm-plugin \
    && uv pip install ."

# MiniMax-H3 CI branch only (zni/h3-ci-unified; not for main):
# tt-vllm-plugin pins transformers==4.55.0, which predates the Qwen3-VL video processor the H3
# pipeline loads for Ref2VA video references (AutoVideoProcessor on
# text_encoder/video_preprocessor_config.json), so every request with a video reference failed
# with "Unrecognized video processor". Restore the transformers pin of the tt-metal this image is
# built on and check the class resolves. The vLLM runners in this image expect the older pin, so
# this image is for MiniMax-H3 only.
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && pin=\$(grep -E '^transformers[[:space:]]*==' ${TT_METAL_HOME}/tt_metal/python_env/requirements-dev.txt | tr -d ' ') \
    && test -n \"\$pin\" \
    && uv pip install --no-cache-dir \"\$pin\" \
    && python -c 'import transformers; from transformers.models.qwen3_vl import Qwen3VLVideoProcessor; print(\"transformers\", transformers.__version__, Qwen3VLVideoProcessor.__name__)'"

# Create separate audio venv for whisperx/pyannote (torch 2.7.x)
# This avoids torch version conflicts with vLLM 0.18.1 which requires torch>=2.10
ENV AUDIO_VENV_DIR=${TT_METAL_HOME}/audio_venv
ENV AUDIO_VENV_PYTHON=${AUDIO_VENV_DIR}/bin/python
RUN /bin/bash -c "python3 -m venv ${AUDIO_VENV_DIR} \
    && source ${AUDIO_VENV_DIR}/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && pip install uv \
    && cd ${TT_METAL_HOME} \
    && uv pip install --no-cache-dir --index-strategy unsafe-best-match --overrides uv-overrides.txt -r audio-requirements.txt \
    && echo 'Audio venv created successfully with whisperx and torch 2.7.x'"

# Fix GPG keys before installing cpp_server dependencies
RUN apt-get clean && rm -rf /var/lib/apt/lists/* \
    && apt-get update --allow-insecure-repositories || true \
    && apt-get install -y --allow-unauthenticated gnupg2 ca-certificates \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4 || true \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 15CF4D18AF4F7421 || true

# cpp_server deps (apt + Rust if needed + Drogon)
RUN cd ${TT_METAL_HOME}/server/cpp_server && ./install_dependencies.sh

# Build C++ server (requires Rust for tokenizers-cpp and Clang-20)
#
# The build tree is pruned in this same RUN. build/ is ~2GB, of which only the
# 79MB tt_media_server_cpp binary is used: run_cpp.sh executes it and nothing else,
# and the binary's RUNPATH is build_Release/lib:/usr/local/lib:/usr/local/share/uv/...
# -- it never loads anything from its own build directory. The rest is build output:
# object files, static archives already linked into the binary, FetchContent sources,
# and ~30 test/benchmark executables. Pruning here (rather than in a later RUN) keeps
# those bytes out of the layer entirely, so it lowers both image size and peak build disk.
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
    && export CARGO_HOME=${HOME_DIR}/.cargo \
    && export RUSTUP_HOME=${HOME_DIR}/.rustup \
    && export PATH=${HOME_DIR}/.cargo/bin:${PATH} \
    && export CXX=clang++-20 \
    && export CC=clang-20 \
    && cd ${TT_METAL_HOME}/server/cpp_server \
    && ./build.sh \
    && rm -rf build/CMakeFiles build/_deps \
    && find build -maxdepth 1 -type f -name '*_test' -delete \
    && find build -maxdepth 1 -type f -name '*_benchmark' -delete \
    && find build -maxdepth 1 -type f -name '*.a' -delete \
    && find build -name '*.o' -delete \
    && test -x build/tt_media_server_cpp"

# Add script for running C++ server
COPY "tt-media-server/run_cpp.sh" "${TT_METAL_HOME}/server/run_cpp.sh"
RUN chmod +x ${TT_METAL_HOME}/server/run_cpp.sh

# Cleanup build-only directories.
#
# Only content that is still needed *above* this point stays here; everything with no
# later consumer is now reclaimed inside the RUN that creates it (see the tt-metal build).
# Do not move these earlier:
#   .cargo / .rustup / /usr/local/rustup  tokenizers-cpp needs Rust during the cpp_server build
#   tt-metal/cmake                        cpp_server/build.sh reads the clang-20 toolchain file from it
#   /usr/include                          every C++ compile above needs it
#   tt-metal/server/*                     only exists after the COPY that follows the tt-metal build
RUN rm -rf ${HOME_DIR}/.rustup \
    && rm -rf ${HOME_DIR}/.cargo \
    && rm -rf ${HOME_DIR}/.cache \
    && rm -rf ${HOME_DIR}/tt-metal/.github \
    && rm -rf ${HOME_DIR}/tt-metal/cmake \
    && rm -rf ${HOME_DIR}/tt-metal/scripts \
    && rm -rf ${HOME_DIR}/tt-metal/server/tests \
    && rm -rf ${HOME_DIR}/tt-metal/server/scripts \
    && rm -rf /usr/local/rustup \
    && rm -rf /usr/include \
    && echo "=== builder home size breakdown (informs the runtime-stage layer split) ===" \
    && du -xsh ${HOME_DIR}/.[!.]* ${HOME_DIR}/* 2>/dev/null | sort -h \
    && du -xsh ${HOME_DIR}/tt-metal/.[!.]* ${HOME_DIR}/tt-metal/* 2>/dev/null | sort -h | tail -15 \
    && du -xsh ${PYTHON_ENV_DIR}/lib/python*/site-packages/* 2>/dev/null | sort -h | tail -12


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
ARG TT_METAL_COMMIT_SHA_OR_TAG=915d9e67ad572b97fbaf5339bd3e518db6dc05e3
ARG CONTAINER_APP_UID=1000
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}

ENV TT_METAL_COMMIT_SHA_OR_TAG=${TT_METAL_COMMIT_SHA_OR_TAG} \
    ENVIRONMENT=development \
    TT_MM_THROTTLE_PERF=5 \
    SHELL=/bin/bash \
    CONTAINER_APP_USERNAME=${CONTAINER_APP_USERNAME} \
    TT_METAL_HOME=${HOME_DIR}/tt-metal \
    TT_METAL_RUNTIME_ROOT=${HOME_DIR}/tt-metal \
    CONFIG=Release \
    TT_METAL_ENV=dev \
    LOGURU_LEVEL=INFO \
    PYTHONPATH=${HOME_DIR}/tt-metal \
    PYTHON_ENV_DIR=${HOME_DIR}/tt-metal/python_env \
    AUDIO_VENV_PYTHON=${HOME_DIR}/tt-metal/audio_venv/bin/python \
    LD_LIBRARY_PATH=${HOME_DIR}/tt-metal/build/lib \
    VLLM_USE_V1=1 \
    TT_METAL_LOGS_PATH=${HOME_DIR}/logs

# Install runtime dependencies
# These packages are required for the application to run in production
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gosu \
    libjsoncpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -u ${CONTAINER_APP_UID} -s /bin/bash -d ${HOME_DIR} ${CONTAINER_APP_USERNAME} \
    && mkdir -p ${HOME_DIR} ${HOME_DIR}/logs \
    && chown -R ${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} ${HOME_DIR} \
    && mkdir -p /run/sshd \
    && chmod 755 /run/sshd

# Copy the builder's home into the runtime image as four layers of roughly equal
# size, instead of one ~16GB layer.
#
# BuildKit compresses each layer on one core at export time, so the single-layer
# copy made "exporting layers" take ~17 min in CI. Layers compress in parallel, so
# the export takes about as long as the largest layer. Sizes measured on the
# 2026-09 image (builder home 16GB): python_env 9.9GB, of which torch 1.6GB,
# nvidia (torch's CUDA runtime) 2.7GB and vllm 1.2GB; build_Release 2.6GB;
# audio_venv 1.6GB; everything else 1.3GB. Grouped along those folders:
#
#   1. python_env: torch + nvidia                                 ~4.3GB
#   2. python_env: everything else (bin/, the other packages)      ~4.4GB
#   3. build_Release + audio_venv                                  ~4.2GB
#   4. python_env: vllm + the rest of the home dir
#      (tt-metal sources, runtime/, server/, models/, .bashrc)     ~2.5GB
#
# Each RUN bind-mounts the builder's home read-only at /mnt/builder and copies its
# share. Ownership is set in the same RUN that writes the files on purpose: a later
# `chown -R` / `chmod -R` would copy every touched file into yet another layer.
# The final filesystem is identical to the single COPY.
ARG APP_OWNER=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME}
ENV SITE_PACKAGES_REL=lib/python3.10/site-packages

# 1. torch and its CUDA runtime libraries
RUN --mount=type=bind,from=builder,source=/home/container_app_user,target=/mnt/builder \
    SP=/mnt/builder/tt-metal/python_env/${SITE_PACKAGES_REL} && test -d "$SP/torch" \
    && mkdir -p ${PYTHON_ENV_DIR}/${SITE_PACKAGES_REL} \
    && cp -a "$SP/torch" "$SP/nvidia" ${PYTHON_ENV_DIR}/${SITE_PACKAGES_REL}/ \
    && chown -R ${APP_OWNER} ${PYTHON_ENV_DIR}

# 2. the rest of the main venv
RUN --mount=type=bind,from=builder,source=/home/container_app_user,target=/mnt/builder \
    tar -C /mnt/builder/tt-metal/python_env --owner=${CONTAINER_APP_USERNAME} --group=${CONTAINER_APP_USERNAME} --anchored \
        --exclude=./${SITE_PACKAGES_REL}/torch --exclude=./${SITE_PACKAGES_REL}/nvidia --exclude=./${SITE_PACKAGES_REL}/vllm \
        -cf - . | tar -C ${PYTHON_ENV_DIR} -xf - \
    && chmod -R +x ${PYTHON_ENV_DIR}/bin

# 3. tt-metal build output and the audio venv
RUN --mount=type=bind,from=builder,source=/home/container_app_user,target=/mnt/builder \
    cp -a /mnt/builder/tt-metal/build_Release /mnt/builder/tt-metal/audio_venv ${TT_METAL_HOME}/ \
    && chown -R ${APP_OWNER} ${TT_METAL_HOME}/build_Release ${TT_METAL_HOME}/audio_venv

# 4. vllm plus everything else in the home dir
RUN --mount=type=bind,from=builder,source=/home/container_app_user,target=/mnt/builder \
    tar -C /mnt/builder --owner=${CONTAINER_APP_USERNAME} --group=${CONTAINER_APP_USERNAME} --anchored \
        --exclude=./tt-metal/python_env --exclude=./tt-metal/audio_venv --exclude=./tt-metal/build_Release \
        -cf - . | tar -C ${HOME_DIR} -xf - \
    && cp -a /mnt/builder/tt-metal/python_env/${SITE_PACKAGES_REL}/vllm ${PYTHON_ENV_DIR}/${SITE_PACKAGES_REL}/ \
    && chown -R ${APP_OWNER} ${PYTHON_ENV_DIR}/${SITE_PACKAGES_REL}/vllm \
    && test -L ${TT_METAL_HOME}/build && test -x ${PYTHON_ENV_DIR}/bin/python

USER ${CONTAINER_APP_USERNAME}
WORKDIR ${HOME_DIR}/tt-metal

# Set up bashrc
RUN echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc

# Switch back to root for entrypoint (it will drop to non-root user via gosu)
USER root
COPY docker-entrypoint.sh ${HOME_DIR}/tt-metal
RUN chmod +x ${HOME_DIR}/tt-metal/docker-entrypoint.sh
RUN chmod +x ${TT_METAL_HOME}/server/run_uvicorn.sh

EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
# Default to uvicorn
ENV SERVER_MODE=uvicorn
CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && cd ${TT_METAL_HOME}/server/ && if [ \"$SERVER_MODE\" = \"cpp\" ]; then source ./run_cpp.sh; else source ./run_uvicorn.sh; fi"]
