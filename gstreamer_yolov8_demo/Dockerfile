# GStreamer YOLOv8s Demo on Tenstorrent
# Customer runs: docker run -p 8080:8080 ... tt-gstreamer-yolov8 stream-test

# Base image with tt-metal and YOLOv8s model
# Use local image or pull from registry:
#   docker build --build-arg BASE_IMAGE=ghcr.io/tenstorrent/tt-inference-server/tt-metal-yolov8s-base:v0.0.1 ...
ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-inference-server/tt-metal-yolov8s-base:v0.0.1
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
        graphviz numpy_ringbuffer opencv-python-headless pytest ultralytics datasets websockets

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

# Copy entrypoint and server scripts
COPY --chown=container_app_user:container_app_user entrypoint.sh /app/entrypoint.sh
COPY --chown=container_app_user:container_app_user http_stream.py /app/http_stream.py
COPY --chown=container_app_user:container_app_user test_8device_parallel.py /app/test_8device_parallel.py
COPY --chown=container_app_user:container_app_user websocket_server.py /app/websocket_server.py
COPY --chown=container_app_user:container_app_user gst_server.py /app/gst_server.py
COPY --chown=container_app_user:container_app_user parallel_8device_stream.py /app/parallel_8device_stream.py

# Copy demo videos (included in image - no external files needed)
COPY --chown=container_app_user:container_app_user city_traffic.mp4 /app/demo/city_traffic.mp4
COPY --chown=container_app_user:container_app_user sample_traffic.mp4 /app/demo/sample_traffic.mp4
RUN chmod +x /app/entrypoint.sh

WORKDIR /home/container_app_user/tt-metal
EXPOSE 8080 8765

ENTRYPOINT ["/app/entrypoint.sh"]
