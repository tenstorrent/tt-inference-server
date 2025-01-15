# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# set with --build-arg TT_METAL_DOCKERFILE_VERSION=<version>
# NOTE: tt-metal Ubuntu 22.04 Dockerfile must be built locally until release images are published
ARG TT_METAL_DOCKERFILE_VERSION

FROM local/tt-metal/tt-metalium/ubuntu-22.04-amd64:$TT_METAL_DOCKERFILE_VERSION

# include shared instructions
COPY vllm.llama3.src.shared.Dockerfile /vllm.llama3.src.shared.Dockerfile
RUN cat /vllm.llama3.src.shared.Dockerfile | docker build - < .
