# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Multi-host extension for distributed vLLM inference (Dual/Quad Galaxy)
# Extends the base vLLM image with sshd for MPI worker processes
#
# Build:
#   docker build -t vllm-multihost \
#     --build-arg BASE_IMAGE=ghcr.io/.../vllm-tt-metal-src-release:TAG \
#     -f vllm.tt-metal.src.multihost.Dockerfile .
#
# Usage:
#   Controller: docker run --user root -e MULTIHOST_ROLE=controller \
#                 -v /path/to/ssh_config:/tmp/ssh_config:ro \
#                 --entrypoint /usr/local/bin/multihost_entrypoint.sh <image> <command>
#   Worker:     docker run --user root -e MULTIHOST_ROLE=worker \
#                 -v /path/to/key.pub:/tmp/authorized_keys.pub:ro \
#                 --entrypoint /usr/local/bin/multihost_entrypoint.sh <image>

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Multi-host support: sshd for MPI worker processes
# - Port 2200 to avoid conflict with host SSH
# - SSH directory prepared for runtime mount of authorized_keys
# - MPI config directory prepared for runtime mount of rankfile
# - Worker entrypoint handles authorized_keys setup and starts sshd
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    gosu \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd \
    && ssh-keygen -A \
    && mkdir -p /home/container_app_user/.ssh \
    && chmod 700 /home/container_app_user/.ssh \
    && chown container_app_user:container_app_user /home/container_app_user/.ssh \
    && mkdir -p /etc/mpirun \
    && echo "Port 2200" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication no" >> /etc/ssh/sshd_config \
    && echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config

# Unified entrypoint script for Worker/Controller SSH setup
COPY vllm-tt-metal/multihost_entrypoint.sh /usr/local/bin/multihost_entrypoint.sh
RUN chmod +x /usr/local/bin/multihost_entrypoint.sh

USER container_app_user
