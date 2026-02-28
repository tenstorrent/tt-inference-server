# wan2.2-t2v-a14b-diffusers Tenstorrent Support on WH Galaxy

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other video models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`wan2.2-t2v-a14b-diffusers` is also supported on hardware:

- [WH LoudBox/QuietBox](wan2.2-t2v-a14b-diffusers_t3k.md)

## Quickstart - Deploy wan2.2-t2v-a14b-diffusers Inference Server on WH Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model wan2.2-t2v-a14b-diffusers --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/65718bb/models/tt_transformers) |
| tt-metal Commit | `65718bb` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.9.0-65718bb` |
