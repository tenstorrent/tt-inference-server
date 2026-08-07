# yolox_nano Tenstorrent Support on N150

#### Useful links

- [N150 details](https://tenstorrent.com/hardware/wormhole)
- [Search other cnn models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`yolox_nano` is also supported on hardware:

- [P150](yolox_nano_p150.md)

## Quickstart - Deploy yolox_nano Inference Server on n150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server (forge plugin)](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model yolox_nano --device n150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [yolox_nano](https://huggingface.co/yolox_nano) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c49bb76/models/tt_transformers) |
| tt-metal Commit | `c49bb76` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server-forge:0.18.0-c49bb76` |
