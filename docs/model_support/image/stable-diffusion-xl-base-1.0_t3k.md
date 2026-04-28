# stable-diffusion-xl-base-1.0 Tenstorrent Support on WH LoudBox/QuietBox

Supported weights variants for this model implementation are:

- `stable-diffusion-xl-base-1.0`: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) **(default)** 
- `stable-diffusion-xl-base-1.0-img-2-img`: [stabilityai/stable-diffusion-xl-base-1.0-img-2-img](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0-img-2-img)

To use non-default weights, replace `stable-diffusion-xl-base-1.0` in commands below.

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other image models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`stable-diffusion-xl-base-1.0` is also supported on hardware:

- [WH Galaxy](stable-diffusion-xl-base-1.0_galaxy.md)
- [BH LoudBox](stable-diffusion-xl-base-1.0_p150x8.md)
- [BH 4xP150](stable-diffusion-xl-base-1.0_p150x4.md)
- [P100/P150](stable-diffusion-xl-base-1.0_p100.md)
- [N150/N300](stable-diffusion-xl-base-1.0_n150.md)

## Quickstart - Deploy stable-diffusion-xl-base-1.0 Inference Server on WH LoudBox/QuietBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-base-1.0 --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [stabilityai/stable-diffusion-xl-base-1.0-img-2-img](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0-img-2-img) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 4 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a8c5af0/models/tt_transformers) |
| tt-metal Commit | `a8c5af0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.11.1-a8c5af0` |
