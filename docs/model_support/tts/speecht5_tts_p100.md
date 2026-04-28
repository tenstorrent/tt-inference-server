# speecht5_tts Tenstorrent Support on P100/P150

#### Useful links

- [P100/P150 details](https://tenstorrent.com/hardware/blackhole)
- [Search other tts models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`speecht5_tts` is also supported on hardware:

- [N150/N300](speecht5_tts_n150.md)

## Quickstart - Deploy speecht5_tts Inference Server on p150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model speecht5_tts --device p150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [speecht5-tts](https://github.com/tenstorrent/tt-metal/tree/2508216/models/experimental/speecht5_tts) |
| tt-metal Commit | `2508216` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.10.0-2508216` |
