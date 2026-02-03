# BGE-Large-EN-v1.5 Tenstorrent Support on Tenstorrent Galaxy

BGE-Large-EN-v1.5 is also supported on:

- [n150](BGE-Large-EN-v1.5_n150.md)
- [n300](BGE-Large-EN-v1.5_n300.md)
- [TT-LoudBox](BGE-Large-EN-v1.5_t3k.md)

#### Back links

- [Tenstorrent Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other embedding models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy BGE-Large-EN-v1.5 Inference Server on Tenstorrent Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model BGE-Large-EN-v1.5 --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Implementation Code | [tt-vllm-plugin](https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-vllm-plugin/tree/2496be4/tt_vllm_plugin) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.2.0-2496be4518bca0a7a5b497a4cda3cfe7e2f59756` |
