# speecht5_tts Tenstorrent Support on n300

speecht5_tts is also supported on:

- [n150](speecht5_tts_n150.md)

#### Back links

- [n300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other tts models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy speecht5_tts Inference Server on n300

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model speecht5_tts --device n300 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [speecht5-tts](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/experimental/speecht5_tts) |
| tt-metal Commit | `a9b09e0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-a9b09e0` |
