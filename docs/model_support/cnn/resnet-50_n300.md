# resnet-50 Tenstorrent Support on N300

#### Useful links

- [N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other cnn models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`resnet-50` is also supported on hardware:

- [N150](resnet-50_n150.md)

## Quickstart - Deploy resnet-50 Inference Server on n300

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server (forge plugin)](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model resnet-50 --device n300 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [resnet-50](https://huggingface.co/resnet-50) |
| Model Status | 🟡 Functional |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/079a2c2/models/tt_transformers) |
| tt-metal Commit | `079a2c2` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server-forge:0.13.0-079a2c2` |
