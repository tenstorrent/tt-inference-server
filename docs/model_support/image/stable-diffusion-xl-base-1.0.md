# stable-diffusion-xl-base-1.0 Tenstorrent Support

## Run stable-diffusion-xl-base-1.0 on n150

[Image Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `stable-diffusion-xl-base-1.0`, the following weights are supported as well:

- `stable-diffusion-xl-base-1.0-img-2-img`

To use these weights simply swap `stable-diffusion-xl-base-1.0` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-base-1.0 --device n150 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [stabilityai/stable-diffusion-xl-base-1.0-img-2-img](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0-img-2-img) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/tt_transformers) |
| tt-metal Commit | `a9b09e0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-a9b09e0` |

## Run stable-diffusion-xl-base-1.0 on n300

[Image Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `stable-diffusion-xl-base-1.0`, the following weights are supported as well:

- `stable-diffusion-xl-base-1.0-img-2-img`

To use these weights simply swap `stable-diffusion-xl-base-1.0` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-base-1.0 --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [stabilityai/stable-diffusion-xl-base-1.0-img-2-img](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0-img-2-img) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/tt_transformers) |
| tt-metal Commit | `a9b09e0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-a9b09e0` |

## Run stable-diffusion-xl-base-1.0 on TT-LoudBox

[Image Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `stable-diffusion-xl-base-1.0`, the following weights are supported as well:

- `stable-diffusion-xl-base-1.0-img-2-img`

To use these weights simply swap `stable-diffusion-xl-base-1.0` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-base-1.0 --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [stabilityai/stable-diffusion-xl-base-1.0-img-2-img](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0-img-2-img) |
| Model Status | 🟢 Complete |
| Max Batch Size | 4 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/tt_transformers) |
| tt-metal Commit | `a9b09e0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-a9b09e0` |

## Run stable-diffusion-xl-base-1.0 on Tenstorrent Galaxy

[Image Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `stable-diffusion-xl-base-1.0`, the following weights are supported as well:

- `stable-diffusion-xl-base-1.0-img-2-img`

To use these weights simply swap `stable-diffusion-xl-base-1.0` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-base-1.0 --device galaxy --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [stabilityai/stable-diffusion-xl-base-1.0-img-2-img](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0-img-2-img) |
| Model Status | 🟢 Complete |
| Max Batch Size | 32 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/tt_transformers) |
| tt-metal Commit | `a9b09e0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-a9b09e0` |
