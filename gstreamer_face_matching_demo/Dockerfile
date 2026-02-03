# GStreamer Face Recognition Demo on Tenstorrent
# YuNet (Face Detection) + SFace (Face Recognition)
#
# Usage:
#   docker build -t face-matching-demo .
#   docker run --rm --network host --device /dev/tenstorrent:/dev/tenstorrent \
#     --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
#     face-matching-demo
#
# Base image: tt-metal with commit 3d6d9a05 (has YuNet in models/experimental/)
# SFace needs to be copied from tvardhineni/sface_model_bringup branch

ARG BASE_IMAGE=tt-metal-base:3d6d9a05
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

# Create directories
RUN mkdir -p /app/gstreamer_plugins/plugins/python /app/models /app/demo

# Copy GStreamer plugin
COPY plugins/yunet_sface.py /app/gstreamer_plugins/plugins/python/

# Create new venv with --system-site-packages for GStreamer access
WORKDIR /home/container_app_user/tt-metal
RUN python3 -m venv /home/container_app_user/python_env_gst --system-site-packages

# Install all required dependencies
RUN /home/container_app_user/python_env_gst/bin/pip install --upgrade pip && \
    /home/container_app_user/python_env_gst/bin/pip install \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    /home/container_app_user/python_env_gst/bin/pip install \
        graphviz numpy_ringbuffer opencv-python-headless pytest websockets onnx

# Install tt-metal
RUN /home/container_app_user/python_env_gst/bin/pip install -e .

# Create __init__.py files for imports
RUN touch /home/container_app_user/tt-metal/tests/__init__.py && \
    touch /home/container_app_user/tt-metal/tests/ttnn/__init__.py

# YuNet is already in base image at models/experimental/yunet/
# Clone YUNet reference repo (github.com/jahongir7174/YUNet) via setup.sh
RUN cd /home/container_app_user/tt-metal/models/experimental/yunet && bash setup.sh

# Copy SFace model (not in base image)
COPY --chown=container_app_user:container_app_user models/sface /home/container_app_user/tt-metal/models/experimental/sface/

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
COPY --chown=container_app_user:container_app_user websocket_server.py /app/websocket_server.py
COPY --chown=container_app_user:container_app_user parallel_8device_stream.py /app/parallel_8device_stream.py

# Copy demo videos if available (optional)
# Copy demo assets including test video
COPY --chown=container_app_user:container_app_user demo/ /app/demo/

RUN chmod +x /app/entrypoint.sh

WORKDIR /home/container_app_user/tt-metal
EXPOSE 8080 8765

ENTRYPOINT ["/app/entrypoint.sh"]
