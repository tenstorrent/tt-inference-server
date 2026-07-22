# Falcon3-7B-Instruct Tenstorrent Support on P150

#### Useful links

- [P150 details](https://tenstorrent.com/hardware/blackhole)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Falcon3-7B-Instruct Inference Server on p150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server (forge plugin)](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Falcon3-7B-Instruct --device p150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [tiiuae/Falcon3-7B-Instruct](https://huggingface.co/tiiuae/Falcon3-7B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 4 |
| Max Context Length | 16384 |
| Implementation Code | [forge-vllm-plugin](https://github.com/tenstorrent/tt-xla/tree/main/tree/c49bb76/integrations/vllm_plugin) |
| tt-metal Commit | `c49bb76` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server-forge:0.18.0-c49bb76` |
