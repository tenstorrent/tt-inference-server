# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# ==============================================================================
# tt-lab-glm launcher image (impl tt_lab_glm, GLM-5.3-Flash)
#
# glm-tt is a prebuilt binary with its own loader and C/C++ libraries, shipped in
# the weights repo (hf_weights_repo) beside the TTQ weights, so this image holds
# neither: only python3, the `hf` CLI, run_glm_tt.py and glm_tt_entrypoint.py.
# The entrypoint downloads the weights repo into $CACHE_ROOT/weights/$MODEL (the
# Helm chart's cache hostPath) and then runs run_glm_tt.py, which verifies the
# package and starts `glm-tt serve` on port 8000.
#
# Build from the repo root:
#   docker build -f vllm-tt-metal/tt-lab-glm.Dockerfile \
#     -t ghcr.io/tenstorrent/tt-inference-server/tt-lab-glm-ubuntu-22.04-amd64:<VERSION>-<commit> .
# ==============================================================================

FROM ubuntu:22.04

LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server
LABEL org.opencontainers.image.description="tt-lab-glm launcher: downloads the weights repo and runs glm-tt serve"

ARG DEBIAN_FRONTEND=noninteractive
# The Helm chart's fix-cache-permissions initContainer chowns the cache to 1000:1000.
ARG CONTAINER_APP_UID=1000
ARG CONTAINER_APP_USERNAME=container_app_user
ARG HOME_DIR=/home/${CONTAINER_APP_USERNAME}
ARG APP_DIR=${HOME_DIR}/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/*

# The `hf` CLI for the weights download, pinned as run.py's host setup pins it.
COPY requirements/hf-setup.txt requirements/constraints.txt /tmp/requirements/
RUN python3 -m venv /opt/hf \
    && /opt/hf/bin/pip install --no-cache-dir -r /tmp/requirements/hf-setup.txt \
    && rm -rf /tmp/requirements

RUN useradd -u ${CONTAINER_APP_UID} -s /bin/bash -d ${HOME_DIR} -m ${CONTAINER_APP_USERNAME}

COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} \
    vllm-tt-metal/src/run_glm_tt.py vllm-tt-metal/src/glm_tt_entrypoint.py \
    ${APP_DIR}/src/
COPY --chown=${CONTAINER_APP_USERNAME}:${CONTAINER_APP_USERNAME} VERSION ${APP_DIR}/VERSION

USER ${CONTAINER_APP_USERNAME}

# GLM_TT_WEIGHTS_REPO is the spec's hf_weights_repo (tests/test_glm53_flash_spec.py checks it).
ENV PATH=/opt/hf/bin:${PATH} \
    CACHE_ROOT=${HOME_DIR}/cache_root \
    GLM_TT_WEIGHTS_REPO=tenstorrent/GLM-5.3-Flash-TTQ \
    PYTHONUNBUFFERED=1

RUN mkdir -p ${CACHE_ROOT}

WORKDIR ${APP_DIR}/src
EXPOSE 8000

# tini reaps the staged glm-tt's children and passes SIGTERM on (the first drains).
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/bin/python3", "/home/container_app_user/app/src/glm_tt_entrypoint.py"]
