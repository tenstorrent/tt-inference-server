# FLUX.1-schnell Tenstorrent Support on BH QuietBox 2

#### Useful links

- [BH QuietBox 2 details](https://tenstorrent.com/hardware/tt-quietbox)
- [Search other image models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy FLUX.1-schnell Inference Server on BH QuietBox 2

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model FLUX.1-schnell --device p300x2 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/555f240/models/tt_transformers) |
| tt-metal Commit | `555f240` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.10.1-555f240` |
