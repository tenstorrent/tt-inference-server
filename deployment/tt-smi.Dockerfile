# =========================================================================
# Stage 1: The "Builder" stage
#
# This stage installs python tools, creates a virtual environment, and uses
# pip to install tt-smi and its dependencies into the venv.
# =========================================================================
FROM ubuntu:22.04 AS builder

# Define a build argument for the tt-smi version. Defaults to empty for "latest".
ARG TT_SMI_VERSION=""

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and the venv tool
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment in a standard location
RUN python3 -m venv /opt/venv

# Install tt-smi: either a specific version or the latest from PyPI
# This shell logic checks if TT_SMI_VERSION is set.
RUN if [ -z "${TT_SMI_VERSION}" ]; then \
    echo "No TT_SMI_VERSION specified, installing latest..."; \
    /opt/venv/bin/pip install --no-cache-dir tt-smi; \
    else \
    echo "Installing specified TT_SMI_VERSION: ${TT_SMI_VERSION}"; \
    /opt/venv/bin/pip install --no-cache-dir tt-smi==${TT_SMI_VERSION}; \
    fi

# =========================================================================
# Stage 2: The "Final" lightweight image
#
# This stage starts fresh and copies the populated venv from the builder.
# It only installs the minimal Python runtime and necessary shared libraries.
# =========================================================================
FROM ubuntu:22.04

LABEL maintainer="Benjamin Goel <bgoel@tenstorrent.com>" \
    org.opencontainers.image.source=https://github.com/tenstorrent/tt-inference-server

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install ONLY the runtime dependencies:
# - python3: The standard python interpreter with a full standard library.
# - libpci3 & libyaml-cpp0.7: Shared libraries required by tt-smi.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    libpci3 \
    libyaml-cpp0.7 \
    ipmitool \
    sudo && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Add the virtual environment's bin directory to the PATH.
# This makes the `tt-smi` command available directly.
ENV PATH="/opt/venv/bin:$PATH"

# Set the default command to be tt-smi
ENTRYPOINT ["tt-smi"]

# Set a default command to run if no arguments are provided to `docker run`
CMD ["--help"]
