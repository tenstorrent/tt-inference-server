# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Optimized multi-stage build for significantly smaller runtime images
ARG TT_METAL_DOCKERFILE_URL

# ==============================================================================
# BUILDER STAGE - Contains all build dependencies and artifacts
# ==============================================================================
FROM ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-cloud-ubuntu-22.04-amd64:0.7.0-71c4d61619ae884adfd4265b25435bf41bb3febf-a186bf4-60472439244 AS builder

# Copy uv's Python installation (venv symlinks point to /root/.local/share/uv/python)
# Make parent directories traversable and uv directory fully accessible

FROM ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-cloud-ubuntu-22.04-amd64:0.7.0-71c4d61619ae884adfd4265b25435bf41bb3febf-a186bf4-60472439244 AS runtime
COPY --from=builder /root/.local/share/uv /root/.local/share/uv
RUN chmod 755 /root /root/.local /root/.local/share && chmod -R 755 /root/.local/share/uv

# Fix venv permissions (COPY --chown can break symlink permissions)
RUN chmod -R +x ${PYTHON_ENV_DIR}/bin

# Switch to non-root user for runtime
USER ${CONTAINER_APP_USERNAME}

# Set working directory
WORKDIR "${APP_DIR}/src"

# Run command
CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && python run_vllm_api_server.py"]
