# Qwen2.5-Coder-32B-Instruct Tenstorrent Support on TT-LoudBox

Qwen2.5-Coder-32B-Instruct is also supported on:

- [Tenstorrent Galaxy (GALAXY_T3K)](Qwen2.5-Coder-32B-Instruct_galaxy_t3k.md)

#### Back links

- [TT-LoudBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Qwen2.5-Coder-32B-Instruct Inference Server on TT-LoudBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Qwen2.5-Coder-32B-Instruct --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/17a5973/models/tt_transformers) |
| tt-metal Commit | `17a5973` |
| vLLM Commit | `aa4ae1e` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-17a5973-aa4ae1e` |
