# bge-m3 Tenstorrent Support on WH LoudBox/QuietBox

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other embedding models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`bge-m3` is also supported on hardware:

- [WH Galaxy](bge-m3_galaxy.md)
- [N150/N300](bge-m3_n150.md)

## Quickstart - Deploy bge-m3 Inference Server on WH LoudBox/QuietBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model bge-m3 --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 4 |
| Implementation Code | [tt-vllm-plugin](https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-vllm-plugin/tree/ec28d12/tt_vllm_plugin) |
| tt-metal Commit | `ec28d12` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.12.0-ec28d12` |
