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

ENV TT_METAL_LOGS_PATH=/home/container_app_user/logs

# Switch back to root for entrypoint
USER root

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && python run_vllm_api_server.py"]
