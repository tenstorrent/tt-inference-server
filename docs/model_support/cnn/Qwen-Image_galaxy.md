# Qwen-Image Tenstorrent Support on Tenstorrent Galaxy

Qwen-Image is also supported on:

- [TT-LoudBox](Qwen-Image_t3k.md)

#### Back links

- [Tenstorrent Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other cnn models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Qwen-Image Inference Server on Tenstorrent Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

The default model weights for this implementation is `Qwen-Image`, the following weights are supported as well:

- `Qwen-Image-2512`

To use these weights simply swap `Qwen-Image` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Qwen-Image --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image), [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) |
| Model Status | 🟡 Functional |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/be88351/models/tt_transformers) |
| tt-metal Commit | `be88351` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-be88351` |
