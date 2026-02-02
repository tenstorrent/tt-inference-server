# Qwen-Image Tenstorrent Support

## Run Qwen-Image on TT-LoudBox

[CNN Model Support Table](../cnn_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Qwen-Image`, the following weights are supported as well:

- `Qwen-Image-2512`

To use these weights simply swap `Qwen-Image` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Qwen-Image --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image), [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) |
| Model Status | 🟡 Functional |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/be88351/models/tt_transformers) |
| tt-metal Commit | `be88351` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-be88351` |

## Run Qwen-Image on Tenstorrent Galaxy

[CNN Model Support Table](../cnn_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Qwen-Image`, the following weights are supported as well:

- `Qwen-Image-2512`

To use these weights simply swap `Qwen-Image` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Qwen-Image --device galaxy --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image), [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) |
| Model Status | 🟡 Functional |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/be88351/models/tt_transformers) |
| tt-metal Commit | `be88351` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-be88351` |
