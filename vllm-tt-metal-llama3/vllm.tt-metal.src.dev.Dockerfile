# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# default base image, override with --build-arg CLOUD_DOCKERFILE_URL=<url or local image path>
ARG CLOUD_DOCKERFILE_URL

FROM ${CLOUD_DOCKERFILE_URL}

# shared build stage, FROM is set by the OS specific Dockerfiles
LABEL maintainer="Tom Stesco <tstesco@tenstorrent.com>"
# connect Github repo with package
LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

# Switch back to root for entrypoint
USER root

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Entrypoint handles venv activation and runs run_vllm_api_server.py
# Usage: docker run <image> --model <hf_repo> --device <device_type>
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
