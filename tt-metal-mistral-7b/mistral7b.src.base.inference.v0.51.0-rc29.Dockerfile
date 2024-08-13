ARG TT_METAL_VERSION=v0.51.0-rc29
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64:$TT_METAL_VERSION-dev

# Build stage
LABEL maintainer="Tom Stesco <tstesco@tenstorrent.com>"

ARG DEBIAN_FRONTEND=noninteractive

ENV TT_METAL_COMMIT_SHA=v0.51.0-rc29
ENV SHELL=/bin/bash
ENV TZ=America/Los_Angeles
# tt-metal build vars
ENV ARCH_NAME=wormhole_b0
ENV TT_METAL_HOME=/tt-metal
ENV CONFIG=Release
ENV TT_METAL_ENV=dev
ENV LOGURU_LEVEL=INFO
# derived vars
ENV PYTHONPATH=${TT_METAL_HOME}
# note: PYTHON_ENV_DIR is used by create_venv.sh
ENV PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env
ENV LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib

# extra system deps
RUN apt-get update && apt-get install -y \
    patchelf \
    libsndfile1 \
    wget \
    nano \
    acl \
    jq \
    vim \
    # user deps
    htop \
    screen \
    tmux \
    unzip \
    zip \
    curl \
    iputils-ping \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# build tt-metal
RUN git clone https://github.com/tenstorrent-metal/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git checkout ${TT_METAL_COMMIT_SHA} \
    && git submodule update --init --recursive \
    && git submodule foreach 'git lfs fetch --all && git lfs pull' \
    && cmake -B build -G Ninja \
    && cmake --build build --target tests \
    && cmake --build build --target install \
    && bash ./create_venv.sh


# Install software-properties-common to manage repositories
# RUN apt-get update && apt-get install -y \
# software-properties-common \
# lsb-release

# Add the deadsnakes PPA and install Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa && \
apt-get update && apt-get install -y python3.9

# user setup
ARG HOME_DIR=/home/user
ARG APP_DIR=tt-metal-mistral-7b
RUN useradd -u 1000 -s /bin/bash -d ${HOME_DIR} user \
    && mkdir -p ${HOME_DIR} \
    && chown -R user:user ${HOME_DIR} \
    && chown -R user:user ${TT_METAL_HOME}

# add user to sudoers
RUN apt-get install -y sudo \
    && echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user \
    && chmod 0440 /etc/sudoers.d/user

# add code-server
EXPOSE 8888
RUN mkdir -p /home/user/.config/code-server/
RUN echo "bind-addr: 0.0.0.0:8888" >> /home/user/.config/code-server/config.yaml \
    && echo "auth: password" >> /home/user/.config/code-server/config.yaml \
    && echo "cert: false" >> /home/user/.config/code-server/config.yaml

ENV CS_VERSION 4.16.1
RUN curl -fOL https://github.com/coder/code-server/releases/download/v${CS_VERSION}/code-server_${CS_VERSION}_amd64.deb && \
    dpkg -i code-server_${CS_VERSION}_amd64.deb && \
    rm code-server_${CS_VERSION}_amd64.deb

USER user

# install app requirements
WORKDIR "${HOME_DIR}/${APP_DIR}"
COPY --chown=user:user "src" "${HOME_DIR}/${APP_DIR}/src"
COPY --chown=user:user "requirements.txt" "${HOME_DIR}/${APP_DIR}/requirements.txt"
RUN /bin/bash -c "source ${PYTHON_ENV_DIR}/bin/activate \
&& pip install --default-timeout=240 --no-cache-dir -r requirements.txt"

RUN echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc

# run app via gunicorn
WORKDIR "${HOME_DIR}/${APP_DIR}/src"
ENV PYTHONPATH=${HOME_DIR}/${APP_DIR}/src:${TT_METAL_HOME}
# CMD ["/bin/bash", "-c", "source ${PYTHON_ENV_DIR}/bin/activate && gunicorn --config gunicorn.conf.py"]

# default port is 7000
ENV SERVICE_PORT=7000
# HEALTHCHECK --retries=5 --start-period=300s CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

ENTRYPOINT code-server