# whisper-large-v3 Tenstorrent Support on TT-LoudBox

whisper-large-v3 is also supported on:

- [n150](whisper-large-v3_n150.md)
- [Tenstorrent Galaxy](whisper-large-v3_galaxy.md)

#### Back links

- [TT-LoudBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other audio models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy whisper-large-v3 Inference Server on TT-LoudBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

The default model weights for this implementation is `whisper-large-v3`, the following weights are supported as well:

- `distil-large-v3`

To use these weights simply swap `whisper-large-v3` for your desired weights in commands below.

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
| Implementation Code | [whisper](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/demos/whisper) |
| tt-metal Commit | `a9b09e0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-a9b09e0` |
