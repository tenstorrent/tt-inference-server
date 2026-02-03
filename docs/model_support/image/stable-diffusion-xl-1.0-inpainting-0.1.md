# stable-diffusion-xl-1.0-inpainting-0.1 Tenstorrent Support

## Run stable-diffusion-xl-1.0-inpainting-0.1 on n150

[Image Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-1.0-inpainting-0.1 --device n150 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/fbbbd2d/models/tt_transformers) |
| tt-metal Commit | `fbbbd2d` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.5.0-fbbbd2da8cfab49ddf43d28dd9c0813a3c3ee2bd` |

## Run stable-diffusion-xl-1.0-inpainting-0.1 on n300

[Image Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-1.0-inpainting-0.1 --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/fbbbd2d/models/tt_transformers) |
| tt-metal Commit | `fbbbd2d` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.5.0-fbbbd2da8cfab49ddf43d28dd9c0813a3c3ee2bd` |

## Run stable-diffusion-xl-1.0-inpainting-0.1 on TT-LoudBox

[Image Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-1.0-inpainting-0.1 --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) |
| Model Status | 🟢 Complete |
| Max Batch Size | 4 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/fbbbd2d/models/tt_transformers) |
| tt-metal Commit | `fbbbd2d` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.5.0-fbbbd2da8cfab49ddf43d28dd9c0813a3c3ee2bd` |

## Run stable-diffusion-xl-1.0-inpainting-0.1 on Tenstorrent Galaxy

[Image Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-1.0-inpainting-0.1 --device galaxy --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) |
| Model Status | 🟢 Complete |
| Max Batch Size | 32 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/fbbbd2d/models/tt_transformers) |
| tt-metal Commit | `fbbbd2d` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.5.0-fbbbd2da8cfab49ddf43d28dd9c0813a3c3ee2bd` |
