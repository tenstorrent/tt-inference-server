# FLUX.1-dev Tenstorrent Support on BH 4xP150

Supported weights variants for this model implementation are:

- `FLUX.1-dev`: [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) **(default)** 
- `FLUX.1-schnell`: [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

To use non-default weights, replace `FLUX.1-dev` in commands below.

#### Useful links

- [BH 4xP150 details](https://tenstorrent.com/hardware/tt-quietbox)
- [Search other image models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`FLUX.1-dev` is also supported on hardware:

- [WH Galaxy](FLUX.1-dev_galaxy.md)
- [BH LoudBox](FLUX.1-dev_p150x8.md)
- [WH LoudBox/QuietBox](FLUX.1-dev_t3k.md)

## Quickstart - Deploy FLUX.1-dev Inference Server on BH 4xP150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model FLUX.1-dev --device p150x4 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a8c5af0/models/tt_transformers) |
| tt-metal Commit | `a8c5af0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.10.0-a8c5af0` |
