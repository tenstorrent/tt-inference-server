# Qwen3-Embedding-4B Tenstorrent Support on WH LoudBox/QuietBox

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other embedding models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen3-Embedding-4B` is also supported on hardware:

- [WH Galaxy](Qwen3-Embedding-4B_galaxy.md)
- [BH QuietBox 2](Qwen3-Embedding-4B_p300x2.md)
- [N150](Qwen3-Embedding-4B_n150.md)
- [N300](Qwen3-Embedding-4B_n300.md)

## Quickstart - Deploy Qwen3-Embedding-4B Inference Server on WH LoudBox/QuietBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server (forge plugin)](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Qwen3-Embedding-4B --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 4 |
| Implementation Code | [forge-vllm-plugin](https://github.com/tenstorrent/tt-xla/tree/main/tree/de59f8a/integrations/vllm_plugin) |
| tt-metal Commit | `de59f8a` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server-forge:0.20.0-de59f8a` |
