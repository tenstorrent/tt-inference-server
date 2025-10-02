# =========================================================================
# Stage 1: The "Builder" stage
#
# This stage installs python tools, creates a virtual environment, and uses
# pip to install tt-smi and its dependencies into the venv.
# =========================================================================
FROM ubuntu:22.04 AS builder

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

# Install tt-smi into the virtual environment
# Using --no-cache-dir keeps this layer smaller
RUN /opt/venv/bin/pip install --no-cache-dir tt-smi

# =========================================================================
# Stage 2: The "Final" lightweight image
#
# This stage starts fresh and copies the populated venv from the builder.
# It only installs the minimal Python runtime and necessary shared libraries.
# =========================================================================
FROM ubuntu:22.04

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
