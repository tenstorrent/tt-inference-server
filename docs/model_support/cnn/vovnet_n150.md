# vovnet Tenstorrent Support on N150/N300

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other cnn models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy vovnet Inference Server on n150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server (forge plugin)](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model vovnet --device n150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [vovnet](https://huggingface.co/vovnet) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/079a2c2/models/tt_transformers) |
| tt-metal Commit | `079a2c2` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server-forge:0.13.0-079a2c2` |

---

## N300 Configuration

### Quickstart - Deploy on n300

**via run.py command**

```bash
python3 run.py --model vovnet --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [vovnet](https://huggingface.co/vovnet) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/079a2c2/models/tt_transformers) |
| tt-metal Commit | `079a2c2` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server-forge:0.13.0-079a2c2` |
