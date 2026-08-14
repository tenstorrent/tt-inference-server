# bge-m3 Tenstorrent Support on BH QuietBox 2

#### Useful links

- [BH QuietBox 2 details](https://tenstorrent.com/hardware/tt-quietbox)
- [Search other embedding models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy bge-m3 Inference Server on BH QuietBox 2

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server (forge plugin)](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model bge-m3 --device p300x2 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 4 |
| Implementation Code | [forge-vllm-plugin](https://github.com/tenstorrent/tt-xla/tree/main/tree/de59f8a/integrations/vllm_plugin) |
| tt-metal Commit | `de59f8a` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server-forge:0.20.0-de59f8a` |
