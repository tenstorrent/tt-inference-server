# mochi-1-preview Tenstorrent Support on BH LoudBox

#### Useful links

- [BH LoudBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other video models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`mochi-1-preview` is also supported on hardware:

- [WH Galaxy](mochi-1-preview_galaxy.md)
- [BH 4xP150](mochi-1-preview_p150x4.md)
- [WH LoudBox/QuietBox](mochi-1-preview_t3k.md)

## Quickstart - Deploy mochi-1-preview Inference Server on BH LoudBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model mochi-1-preview --device p150x8 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [genmo/mochi-1-preview](https://huggingface.co/genmo/mochi-1-preview) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/555f240/models/tt_transformers) |
| tt-metal Commit | `555f240` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.10.0-555f240` |
