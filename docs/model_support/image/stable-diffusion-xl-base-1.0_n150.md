# stable-diffusion-xl-base-1.0 Tenstorrent Support on N150/N300

Supported weights variants for this model implementation are:

- `stable-diffusion-xl-base-1.0`: [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) **(default)** 
- `stable-diffusion-xl-base-1.0-img-2-img`: [stabilityai/stable-diffusion-xl-base-1.0-img-2-img](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0-img-2-img)

To use non-default weights, replace `stable-diffusion-xl-base-1.0` in commands below.

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other image models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`stable-diffusion-xl-base-1.0` is also supported on hardware:

- [WH Galaxy](stable-diffusion-xl-base-1.0_galaxy.md)
- [WH LoudBox/QuietBox](stable-diffusion-xl-base-1.0_t3k.md)

## Quickstart - Deploy stable-diffusion-xl-base-1.0 Inference Server on n150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-base-1.0 --device n150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [stabilityai/stable-diffusion-xl-base-1.0-img-2-img](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0-img-2-img) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/bac8b34/models/tt_transformers) |
| tt-metal Commit | `bac8b34` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.11.1-bac8b34` |

---

## N300 Configuration

### Quickstart - Deploy on n300

**via run.py command**

```bash
python3 run.py --model stable-diffusion-xl-base-1.0 --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [stabilityai/stable-diffusion-xl-base-1.0-img-2-img](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0-img-2-img) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/bac8b34/models/tt_transformers) |
| tt-metal Commit | `bac8b34` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.11.1-bac8b34` |
