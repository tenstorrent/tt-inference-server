# Llama-3.2-3B Tenstorrent Support on n300

Llama-3.2-3B is also supported on:

- [n150](Llama-3.2-3B_n150.md)
- [TT-LoudBox](Llama-3.2-3B_t3k.md)

#### Back links

- [n300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Llama-3.2-3B Inference Server on n300

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

The default model weights for this implementation is `Llama-3.2-3B`, the following weights are supported as well:

- `Llama-3.2-3B-Instruct`

To use these weights simply swap `Llama-3.2-3B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.2-3B --device n300 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B), [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/20edc39/models/tt_transformers) |
| tt-metal Commit | `20edc39` |
| vLLM Commit | `03cb300` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-20edc39-03cb300` |
