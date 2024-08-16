ARG TT_METAL_VERSION=v0.51.0-rc29
FROM ghcr.io/tenstorrent/tt-inference-server/tt-metal-mistral-7b-src-base:v0.0.1-tt-metal-${TT_METAL_VERSION}

# Build stage
LABEL maintainer="Tom Stesco <tstesco@tenstorrent.com>"

ARG DEBIAN_FRONTEND=noninteractive

USER root

# add code-server
EXPOSE 8888
RUN mkdir -p /home/user/.config/code-server/
RUN echo "bind-addr: 0.0.0.0:8888" >> /home/user/.config/code-server/config.yaml \
    && echo "auth: password" >> /home/user/.config/code-server/config.yaml \
    && echo "cert: false" >> /home/user/.config/code-server/config.yaml

ENV CS_VERSION=4.16.1
RUN curl -fOL https://github.com/coder/code-server/releases/download/v${CS_VERSION}/code-server_${CS_VERSION}_amd64.deb && \
    dpkg -i code-server_${CS_VERSION}_amd64.deb && \
    rm code-server_${CS_VERSION}_amd64.deb

USER user
ENTRYPOINT ["code-server"]
