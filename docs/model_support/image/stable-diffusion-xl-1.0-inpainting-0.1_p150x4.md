# stable-diffusion-xl-1.0-inpainting-0.1 Tenstorrent Support on BH 4xP150

#### Useful links

- [BH 4xP150 details](https://tenstorrent.com/hardware/tt-quietbox)
- [Search other image models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`stable-diffusion-xl-1.0-inpainting-0.1` is also supported on hardware:

- [WH Galaxy](stable-diffusion-xl-1.0-inpainting-0.1_galaxy.md)
- [BH LoudBox](stable-diffusion-xl-1.0-inpainting-0.1_p150x8.md)
- [P100/P150](stable-diffusion-xl-1.0-inpainting-0.1_p100.md)
- [WH LoudBox/QuietBox](stable-diffusion-xl-1.0-inpainting-0.1_t3k.md)
- [N150/N300](stable-diffusion-xl-1.0-inpainting-0.1_n150.md)

## Quickstart - Deploy stable-diffusion-xl-1.0-inpainting-0.1 Inference Server on BH 4xP150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-1.0-inpainting-0.1 --device p150x4 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [diffusers/stable-diffusion-xl-1.0-inpainting-0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/fbbbd2d/models/tt_transformers) |
| tt-metal Commit | `fbbbd2d` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.5.0-fbbbd2da8cfab49ddf43d28dd9c0813a3c3ee2bd` |
