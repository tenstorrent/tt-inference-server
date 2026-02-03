# Mistral-7B-Instruct-v0.3 Tenstorrent Support on TT-LoudBox

Mistral-7B-Instruct-v0.3 is also supported on:

- [n150](Mistral-7B-Instruct-v0.3_n150.md)
- [n300](Mistral-7B-Instruct-v0.3_n300.md)

#### Back links

- [TT-LoudBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Mistral-7B-Instruct-v0.3 Inference Server on TT-LoudBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Mistral-7B-Instruct-v0.3 --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |
| Model Status | 🟢 Complete |
| Max Batch Size | 32 |
| Max Context Length | 32768 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/9b67e09/models/tt_transformers) |
| tt-metal Commit | `9b67e09` |
| vLLM Commit | `a91b644` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-9b67e09-a91b644` |
