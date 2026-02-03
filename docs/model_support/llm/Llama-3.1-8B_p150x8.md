# Llama-3.1-8B Tenstorrent Support on 8xp150

Llama-3.1-8B is also supported on:

- [n150](Llama-3.1-8B_n150.md)
- [n300](Llama-3.1-8B_n300.md)
- [TT-LoudBox](Llama-3.1-8B_t3k.md)
- [Tenstorrent Galaxy](Llama-3.1-8B_galaxy.md)
- [Tenstorrent Galaxy (GALAXY_T3K)](Llama-3.1-8B_galaxy_t3k.md)
- [p100](Llama-3.1-8B_p100.md)
- [p150](Llama-3.1-8B_p150.md)
- [4xp150](Llama-3.1-8B_p150x4.md)

#### Back links

- [8xp150 details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Llama-3.1-8B Inference Server on 8xp150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

The default model weights for this implementation is `Llama-3.1-8B`, the following weights are supported as well:

- `Llama-3.1-8B-Instruct`

To use these weights simply swap `Llama-3.1-8B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device p150x8 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 256 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/55fd115/models/tt_transformers) |
| tt-metal Commit | `55fd115` |
| vLLM Commit | `aa4ae1e` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-55fd115-aa4ae1e` |
