# motif-image-6b-preview Tenstorrent Support on WH LoudBox/QuietBox

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other image models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`motif-image-6b-preview` is also supported on:

- [WH Galaxy](motif-image-6b-preview_galaxy.md)

## Quickstart - Deploy motif-image-6b-preview Inference Server on WH LoudBox/QuietBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model motif-image-6b-preview --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Motif-Technologies/Motif-Image-6B-Preview](https://huggingface.co/Motif-Technologies/Motif-Image-6B-Preview) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c180ef7/models/tt_transformers) |
| tt-metal Commit | `c180ef7` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-c180ef7` |
