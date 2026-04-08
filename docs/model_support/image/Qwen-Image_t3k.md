# Qwen-Image Tenstorrent Support on WH LoudBox/QuietBox

Supported weights variants for this model implementation are:

- `Qwen-Image`: [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) **(default)** 
- `Qwen-Image-2512`: [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512)

To use non-default weights, replace `Qwen-Image` in commands below.

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other image models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen-Image` is also supported on hardware:

- [WH Galaxy](Qwen-Image_galaxy.md)

## Quickstart - Deploy Qwen-Image Inference Server on WH LoudBox/QuietBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Qwen-Image --device t3k --workflow server --docker-server
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
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.9.0-be88351` |
