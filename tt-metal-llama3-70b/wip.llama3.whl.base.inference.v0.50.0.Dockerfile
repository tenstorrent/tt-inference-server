FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-20.04-amd64:v0.51.0-rc11-dev

# Build stage
LABEL maintainer="Tom Stesco <tstesco@tenstorrent.com>"

ARG DEBIAN_FRONTEND=noninteractive

# ENV TT_METAL_COMMIT_SHA=a053bc8c9cc380804db730ed7ed084d104abb6a0
ENV SHELL=/bin/bash
ENV TZ=America/Los_Angeles
ENV TT_METAL_HOME=/tt-metal
# ENV ARCH_NAME=wormhole_b0
# ENV CONFIG=Release
# ENV TT_METAL_ENV=dev
# ENV LOGURU_LEVEL=INFO
# derived variables
ENV PYTHONPATH=${TT_METAL_HOME}
# note: PYTHON_ENV_DIR is used by create_venv.sh
# ENV PYTHON_ENV_DIR=${TT_METAL_HOME}/python_env
# ENV LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib

RUN apt-get update && apt-get install -y patchelf


# user setup
ARG HOME_DIR=/home/user
ARG APP_DIR=tt-metal-llama3-70b
RUN useradd -u 1000 -s /bin/bash -d ${HOME_DIR} user \
&& mkdir -p ${HOME_DIR} \
&& mkdir -p ${TT_METAL_HOME} \
&& chown -R user:user ${HOME_DIR} \
&& chown -R user:user ${TT_METAL_HOME}

USER user
WORKDIR "${HOME_DIR}"

RUN wget https://github.com/tenstorrent/tt-metal/releases/download/v0.50.0/metal_libs-0.50.0+wormhole.b0-cp38-cp38-linux_x86_64.whl \
    && pip install metal_libs-0.50.0+wormhole.b0-cp38-cp38-linux_x86_64.whl

# only need model implemenation python source
RUN git clone --depth 1 --branch v0.50.0 https://github.com/tenstorrent/tt-metal.git ${TT_METAL_HOME} \
    && git submodule update --remote models/demos/t3000/llama2_70b/reference/llama

    # install app requirements
WORKDIR "${HOME_DIR}/${APP_DIR}"
COPY --chown=user:user "src" "${HOME_DIR}/${APP_DIR}/src"
COPY --chown=user:user "requirements.txt" "${HOME_DIR}/${APP_DIR}/requirements.txt"
RUN /bin/bash -c "pip install --default-timeout=240 --no-cache-dir -r requirements.txt"

# RUN echo "source ${PYTHON_ENV_DIR}/bin/activate" >> ${HOME_DIR}/.bashrc

# run app via gunicorn
WORKDIR "${HOME_DIR}/${APP_DIR}/src"
ENV PYTHONPATH=${HOME_DIR}/${APP_DIR}/src:${TT_METAL_HOME}
CMD ["/bin/bash", "-c", "gunicorn --config gunicorn.conf.py"]

# default port is 7000
ENV SERVICE_PORT=7000
HEALTHCHECK --retries=5 --start-period=300s CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1
