# Dev Container From A Terminal

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
