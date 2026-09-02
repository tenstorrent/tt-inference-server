# syntax=docker/dockerfile:1.7-labs
# ^ Required for `COPY --exclude` in the runtime stage. A parser directive is only
# honoured before the first comment or instruction, so this MUST stay on line 1.
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
    && CXX=clang++-17 CC=clang-17 bash ./create_venv.sh --bundle-python \
    && echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc \
    && uv pip install --upgrade setuptools wheel \
    && bash ./build_metal.sh --release -e \
    && bash ./build_metal.sh --release -e --build-metal-tests -c --configure-only \
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
    `# Dangling symlinks left by the tt-metal build (e.g. umd/compile_commands.json).` \
    `# Useless at runtime, and COPY --exclude in the runtime stage errors out when` \
    `# one is present - see moby/buildkit#4946. Keep this before that COPY.` \
    && find ${HOME_DIR} -xtype l -exec rm -f {} +


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

# Copy the builder's home in several layers instead of one.
#
# As a single COPY this produced ONE ~15GB layer - 66% of the image, new on every
# build. A throttled `docker pull` from ghcr.io aborts and discards whatever of a
# layer it had in flight, and Docker cannot resume a partial layer, so that one
# layer either arrived in a single unbroken stretch or the whole pull failed.
# tt-shield run 33613146896 lost 5 consecutive attempts to it (38 of 41 layers
# landed in under a minute; the remaining big one never finished).
#
# Ordering is load-bearing: a layer counts as "Already exists" on a pull only if
# everything BENEATH it is unchanged too, so the subtrees that survive a rebuild
# go first. python_env/audio_venv/uv-python change only when the requirements
# files do - a metal-only bump leaves them byte-identical, so runners can reuse
# them instead of re-fetching 9.6GB. Keep volatile subtrees last, and do not
# reorder these without re-reading this comment.
#
# Sizes at the time of writing (uncompressed, from `du -sh` inside the image):
#   python_env 9.6G | build_Release 2.0G | audio_venv 1.6G | runtime 480M
#   server 233M | uv-python 87M | everything else ~250M
#
# --- stable: reused across builds ---
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    /home/container_app_user/tt-metal/uv-python ${TT_METAL_HOME}/uv-python
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    /home/container_app_user/tt-metal/python_env ${PYTHON_ENV_DIR}
# NB: AUDIO_VENV_DIR is set in the builder stage and ENV does not cross stages -
# ${AUDIO_VENV_DIR} would expand to nothing here. Spell it from TT_METAL_HOME.
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    /home/container_app_user/tt-metal/audio_venv ${TT_METAL_HOME}/audio_venv

# --- volatile: rebuilt on every metal / inference-server commit ---
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    /home/container_app_user/tt-metal/build_Release ${TT_METAL_HOME}/build_Release
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    /home/container_app_user/tt-metal/runtime ${TT_METAL_HOME}/runtime
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    /home/container_app_user/tt-metal/server ${TT_METAL_HOME}/server

# --- everything else, minus what the layers above already hold ---
# A catch-all (rather than an enumerated list) so a new directory in tt-metal
# lands in the image on its own instead of silently going missing. The excludes
# are mandatory: without them these subtrees would be stored twice and the image
# would roughly double. `tt-metal/build` is a relative symlink to build_Release
# and rides along here - it resolves because the target is copied above.
COPY --from=builder --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    --exclude=tt-metal/uv-python \
    --exclude=tt-metal/python_env \
    --exclude=tt-metal/audio_venv \
    --exclude=tt-metal/build_Release \
    --exclude=tt-metal/runtime \
    --exclude=tt-metal/server \
    /home/container_app_user ${HOME_DIR}

# Fix venv permissions (COPY --chown can break symlink permissions)
RUN chmod -R +x ${PYTHON_ENV_DIR}/bin

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
