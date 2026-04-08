# whisper-large-v3 Tenstorrent Support on WH LoudBox/QuietBox

Supported weights variants for this model implementation are:

- `whisper-large-v3`: [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) **(default)** 
- `distil-large-v3`: [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)

To use non-default weights, replace `whisper-large-v3` in commands below.

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other audio models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`whisper-large-v3` is also supported on hardware:

- [WH Galaxy](whisper-large-v3_galaxy.md)
- [N150/N300](whisper-large-v3_n150.md)

## Quickstart - Deploy whisper-large-v3 Inference Server on WH LoudBox/QuietBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model whisper-large-v3 --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) |
| Model Status | 🟢 Complete |
| Max Batch Size | 4 |
| Implementation Code | [whisper](https://github.com/tenstorrent/tt-metal/tree/bac8b34/models/demos/whisper) |
| tt-metal Commit | `bac8b34` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.11.1-bac8b34` |
