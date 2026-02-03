# stable-diffusion-xl-base-1.0 Tenstorrent Support on TT-LoudBox

stable-diffusion-xl-base-1.0 is also supported on:

- [n150](stable-diffusion-xl-base-1.0_n150.md)
- [n300](stable-diffusion-xl-base-1.0_n300.md)
- [Tenstorrent Galaxy](stable-diffusion-xl-base-1.0_galaxy.md)

#### Back links

- [TT-LoudBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other image models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy stable-diffusion-xl-base-1.0 Inference Server on TT-LoudBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

The default model weights for this implementation is `stable-diffusion-xl-base-1.0`, the following weights are supported as well:

- `stable-diffusion-xl-base-1.0-img-2-img`

To use these weights simply swap `stable-diffusion-xl-base-1.0` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-base-1.0 --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [stabilityai/stable-diffusion-xl-base-1.0-img-2-img](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0-img-2-img) |
| Model Status | 🟢 Complete |
| Max Batch Size | 4 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/tt_transformers) |
| tt-metal Commit | `a9b09e0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-a9b09e0` |
