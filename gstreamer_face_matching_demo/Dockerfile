# GStreamer Face Recognition Demo on Tenstorrent (Blackhole)
# YuNet (Face Detection) + SFace (Face Recognition)
#
# Base image must be a tt-metal Blackhole image with SFace + YUNet already built (e.g. tt-metal-base-sface:blackhole).
# We do not copy sface/yunet or run setup.sh — everything comes from the base.
#
# Usage (run from gstreamer_face_matching_demo/ so context has plugins/, entrypoint.sh, demo/, etc.):
#   cd gstreamer_face_matching_demo && docker build -f Dockerfile -t face-matching-demo:blackhole .
#   docker run --rm -it --device /dev/tenstorrent -v /dev/hugepages-1G:/dev/hugepages-1G \
#     -v ~/faces:/app/registered_faces -p 8080:8080 -p 8765:8765 face-matching-demo:blackhole

ARG BASE_IMAGE=tt-metal-base-sface:blackhole
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

# Install all required dependencies (incl. ultralytics for YOLOv8s demo)
RUN /home/container_app_user/python_env_gst/bin/pip install --upgrade pip && \
    /home/container_app_user/python_env_gst/bin/pip install \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    /home/container_app_user/python_env_gst/bin/pip install \
        graphviz numpy_ringbuffer opencv-python-headless pytest websockets onnx scikit-learn ultralytics

# Install tt-metal
RUN /home/container_app_user/python_env_gst/bin/pip install -e .

# Create __init__.py files for imports
RUN touch /home/container_app_user/tt-metal/tests/__init__.py && \
    touch /home/container_app_user/tt-metal/tests/ttnn/__init__.py

# SFace, YUNet, and fixes are already in the base image — do not copy or overwrite.

# Set ownership and add container user to video group so /dev/video0 is accessible
RUN chown -R container_app_user:container_app_user /home/container_app_user/python_env_gst /app && \
    (getent group video >/dev/null || groupadd -g 44 video) && usermod -aG video container_app_user

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
COPY --chown=container_app_user:container_app_user benchmark_gst_elements.py /app/benchmark_gst_elements.py
COPY --chown=container_app_user:container_app_user benchmark_lfw.py /app/benchmark_lfw.py
COPY --chown=container_app_user:container_app_user stream_webcam_python.py /app/stream_webcam_python.py

# Copy demo videos if available (optional)
# Copy demo assets including test video and device2 photos
COPY --chown=container_app_user:container_app_user demo/ /app/demo/

# Copy pre-registered faces into the image (Jim, Jonathan Su, Mohamed, Teja)
COPY --chown=container_app_user:container_app_user registered_faces/ /app/registered_faces/

# Generate device 1 & 2 videos from face photos so they're always in the image (skip if already present)
RUN /home/container_app_user/python_env_gst/bin/python /app/demo/create_device_videos_from_faces.py || true

RUN chmod +x /app/entrypoint.sh

WORKDIR /home/container_app_user/tt-metal
EXPOSE 8080 8765

ENTRYPOINT ["/app/entrypoint.sh"]
