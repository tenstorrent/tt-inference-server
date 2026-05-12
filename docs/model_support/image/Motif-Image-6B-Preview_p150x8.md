# Motif-Image-6B-Preview Tenstorrent Support on BH LoudBox

#### Useful links

- [BH LoudBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other image models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Motif-Image-6B-Preview` is also supported on hardware:

- [WH Galaxy](Motif-Image-6B-Preview_galaxy.md)
- [WH LoudBox/QuietBox](Motif-Image-6B-Preview_t3k.md)

## Quickstart - Deploy Motif-Image-6B-Preview Inference Server on BH LoudBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Motif-Image-6B-Preview --device p150x8 --workflow server --docker-server
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
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.9.0-c180ef7` |
