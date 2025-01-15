# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# default base image, override with --build-arg TT_METAL_DOCKERFILE_VERSION=<version>
ARG TT_METAL_DOCKERFILE_VERSION=v0.53.0-rc34

FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64:$TT_METAL_DOCKERFILE_VERSION-dev

# include shared instructions
COPY vllm.llama3.src.shared.Dockerfile /vllm.llama3.src.shared.Dockerfile
RUN cat /vllm.llama3.src.shared.Dockerfile | docker build - < .
