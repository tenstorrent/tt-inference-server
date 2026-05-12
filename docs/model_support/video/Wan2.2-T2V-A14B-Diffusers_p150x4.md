# Wan2.2-T2V-A14B-Diffusers Tenstorrent Support on BH 4xP150

#### Useful links

- [BH 4xP150 details](https://tenstorrent.com/hardware/tt-quietbox)
- [Search other video models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Wan2.2-T2V-A14B-Diffusers` is also supported on hardware:

- [WH Galaxy](Wan2.2-T2V-A14B-Diffusers_galaxy.md)
- [BH LoudBox](Wan2.2-T2V-A14B-Diffusers_p150x8.md)
- [WH LoudBox/QuietBox](Wan2.2-T2V-A14B-Diffusers_t3k.md)

## Quickstart - Deploy Wan2.2-T2V-A14B-Diffusers Inference Server on BH 4xP150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Wan2.2-T2V-A14B-Diffusers --device p150x4 --workflow server --docker-server
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
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.10.1-555f240` |
