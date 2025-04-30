# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# default base image, override with --build-arg CLOUD_DOCKERFILE_URL=<url or local image path>
ARG CLOUD_DOCKERFILE_URL

FROM ${CLOUD_DOCKERFILE_URL}

# Build stage
LABEL maintainer="Ben Goel <bgoel@tenstorrent.com>"
# connect Github repo with package
LABEL org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

# Switch back to root for entrypoint
USER root

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# spinup inference server
CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && pytest ${APP_DIR}/server/gunicorn_app.py -s"]
