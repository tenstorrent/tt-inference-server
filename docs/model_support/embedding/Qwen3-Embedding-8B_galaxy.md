# Qwen3-Embedding-8B Tenstorrent Support on WH Galaxy

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other embedding models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen3-Embedding-8B` is also supported on hardware:

- [WH LoudBox/QuietBox](Qwen3-Embedding-8B_t3k.md)
- [N150/N300](Qwen3-Embedding-8B_n150.md)

## Quickstart - Deploy Qwen3-Embedding-8B Inference Server on WH Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Qwen3-Embedding-8B --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/555f240/models/tt_transformers) |
| tt-metal Commit | `555f240` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.2.0-2496be4518bca0a7a5b497a4cda3cfe7e2f59756` |
