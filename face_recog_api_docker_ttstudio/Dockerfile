# Face recognition REST API on top of Tenstorrent SFace+YuNet image.
# Build:  docker build -t face-recognition-api:local .
# Run:    docker run --rm -it -p 7000:7000 --device /dev/tenstorrent \
#           -v /dev/hugepages-1G:/dev/hugepages-1G \
#           -v $(pwd)/registered_faces:/app/registered_faces \
#           face-recognition-api:local

FROM ghcr.io/tenstorrent/tt-inference-server/tt-metal-sface-yolov8s:blackhole

USER root
RUN mkdir -p /app/registered_faces && chown container_app_user:container_app_user /app/registered_faces

COPY requirements.txt /tmp/face-api-requirements.txt
# tt-metal venv may exist without pip; bootstrap into that same interpreter as entrypoint.sh.
RUN set -eux; \
    PY="/home/container_app_user/tt-metal/python_env/bin/python"; \
    if [ ! -x "$PY" ]; then PY="python3"; fi; \
    if ! "$PY" -m pip --version >/dev/null 2>&1; then \
      apt-get update; \
      apt-get install -y --no-install-recommends curl ca-certificates; \
      curl -sS https://bootstrap.pypa.io/get-pip.py | "$PY"; \
      rm -rf /var/lib/apt/lists/*; \
    fi; \
    "$PY" -m pip install --upgrade pip setuptools wheel; \
    "$PY" -m pip install --no-cache-dir -r /tmp/face-api-requirements.txt

COPY --chown=container_app_user:container_app_user app /home/container_app_user/face-recognition-api/app
COPY --chown=container_app_user:container_app_user entrypoint.sh /home/container_app_user/face-recognition-api/entrypoint.sh
RUN chmod +x /home/container_app_user/face-recognition-api/entrypoint.sh

USER container_app_user
WORKDIR /home/container_app_user/face-recognition-api

ENV FACES_DIR=/home/container_app_user/cache_root/registered_faces
ENV SERVICE_PORT=7070

EXPOSE 7070
ENTRYPOINT ["/home/container_app_user/face-recognition-api/entrypoint.sh"]
