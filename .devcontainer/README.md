# Dev Container From A Terminal

This directory contains a mechanism to create development environment and enable development on bare-metal Tenstorrent machines where we do not have `sudo` access.

The mechanism is based on a enriched `docker run` command, to map the right ports and install the right dependencies.

If you prefer starting from scratch, you can use:
```bash
mkdir $HOME/docker-workspace

docker run -it \
    --device=/dev/tenstorrent \
    -v /dev/shm:/dev/shm \
    --ipc=host \
    -v "$SSH_AUTH_SOCK:/ssh-agent" \
    -e SSH_AUTH_SOCK=/ssh-agent \
    -v "$HOME/.ssh:/home/user/.ssh:ro" \
    --mount source=/dev/hugepages-1G,target=/dev/hugepages-1G,type=bind \
    --mount source="$HOME/docker-workspace",target="/home/workspace",type=bind \
    --entrypoint /bin/bash \
    ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
```

## Setup

Cursor starts this container automatically when the repository is opened from (example):

```bash
/data/<user>/container-dev/tt-inference-server
```

but it can work from any folder that has `<user>` in its name. 

Opening `tt-inference-server` from another parent directory, for example `/data/<user>/tt-inference-server`, is blocked because the container paths are expected to be stable across users and tools.

To run the same development container **without Cursor**:

```bash
cd /data/<user>/container-dev/tt-inference-server
.devcontainer/run-devcontainer.sh
```

If the `tt-metal` build cache is stale, run the same launcher with `--clean`:

```bash
.devcontainer/run-devcontainer.sh --clean
```

This runs `tt-media-server/cpp_server/tt-llm-engine/tt-metal/build_metal.sh --clean` inside the container before installing and rebuilding dependencies.

The script:

- validates that the repository is under `container-dev`;
- builds the local dev image from `.devcontainer/Dockerfile`;
- starts Docker with Tenstorrent device access, hugepages, `/dev/shm`, ports `8000` and `9000`, and the `/data/<user>` mount;
- runs the same dependency setup used by Cursor:

```bash
cd tt-media-server/cpp_server
./install_dependencies.sh
cd tt-llm-engine
./setup.sh --all
cd ..
./build.sh --blaze
```

Set `TT_INFERENCE_DEV_IMAGE` to override the local Docker image tag:

```bash
TT_INFERENCE_DEV_IMAGE=my-tt-inference-dev .devcontainer/run-devcontainer.sh
```
