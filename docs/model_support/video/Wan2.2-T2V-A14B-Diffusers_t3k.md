# Wan2.2-T2V-A14B-Diffusers Tenstorrent Support on WH LoudBox/QuietBox

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other video models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Wan2.2-T2V-A14B-Diffusers` is also supported on hardware:

- [WH Galaxy](Wan2.2-T2V-A14B-Diffusers_galaxy.md)
- [BH LoudBox](Wan2.2-T2V-A14B-Diffusers_p150x8.md)
- [BH 4xP150](Wan2.2-T2V-A14B-Diffusers_p150x4.md)

## Quickstart - Deploy Wan2.2-T2V-A14B-Diffusers Inference Server on WH LoudBox/QuietBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Wan2.2-T2V-A14B-Diffusers --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/555f240/models/tt_transformers) |
| tt-metal Commit | `555f240` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.10.0-555f240` |
