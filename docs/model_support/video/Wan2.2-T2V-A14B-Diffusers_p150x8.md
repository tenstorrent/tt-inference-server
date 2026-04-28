# Wan2.2-T2V-A14B-Diffusers Tenstorrent Support on BH LoudBox

#### Useful links

- [BH LoudBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other video models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Wan2.2-T2V-A14B-Diffusers` is also supported on hardware:

- [WH Galaxy](Wan2.2-T2V-A14B-Diffusers_galaxy.md)
- [BH 4xP150](Wan2.2-T2V-A14B-Diffusers_p150x4.md)
- [WH LoudBox/QuietBox](Wan2.2-T2V-A14B-Diffusers_t3k.md)

## Quickstart - Deploy Wan2.2-T2V-A14B-Diffusers Inference Server on BH LoudBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Wan2.2-T2V-A14B-Diffusers --device p150x8 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a8c5af0/models/tt_transformers) |
| tt-metal Commit | `a8c5af0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.10.0-a8c5af0` |
