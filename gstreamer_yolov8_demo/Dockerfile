# GStreamer YOLOv8s Demo on Tenstorrent
# Customer runs: docker run -p 8080:8080 ... tt-gstreamer-yolov8 stream-test

ARG BASE_IMAGE=sdxl-inf-server_89c6a49:latest
FROM ${BASE_IMAGE}

USER root

# Install GStreamer and Python GI bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-gi \
    python3-gi-cairo \
    python3-gst-1.0 \
    gstreamer1.0-python3-plugin-loader \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

# Create plugin directory
RUN mkdir -p /app/gstreamer_plugins/plugins/python

# Copy GStreamer plugin
COPY plugins/yolov8s.py /app/gstreamer_plugins/plugins/python/

# Create new venv with --system-site-packages for GStreamer access
WORKDIR /home/container_app_user/tt-metal
RUN python3 -m venv /home/container_app_user/python_env_gst --system-site-packages

# Install all required dependencies
RUN /home/container_app_user/python_env_gst/bin/pip install --upgrade pip && \
    /home/container_app_user/python_env_gst/bin/pip install \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    /home/container_app_user/python_env_gst/bin/pip install \
        graphviz numpy_ringbuffer opencv-python-headless pytest ultralytics datasets

# Install tt-metal
RUN /home/container_app_user/python_env_gst/bin/pip install -e .

# Create __init__.py files for imports
RUN touch /home/container_app_user/tt-metal/tests/__init__.py && \
    touch /home/container_app_user/tt-metal/tests/ttnn/__init__.py

# Set ownership
RUN chown -R container_app_user:container_app_user /home/container_app_user/python_env_gst /app

USER container_app_user

# Environment
ENV GST_PLUGIN_PATH=/app/gstreamer_plugins/plugins
ENV PYTHONPATH=/home/container_app_user/tt-metal
ENV GST_DEBUG=1

# Copy entrypoint
COPY --chown=container_app_user:container_app_user entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /home/container_app_user/tt-metal
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
