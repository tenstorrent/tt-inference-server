# Llama-3.3-70B-Instruct Tenstorrent Support on Tenstorrent Galaxy

Llama-3.3-70B-Instruct is also supported on:

- [TT-LoudBox](Llama-3.3-70B-Instruct_t3k.md)
- [Tenstorrent Galaxy (GALAXY_T3K)](Llama-3.3-70B-Instruct_galaxy_t3k.md)
- [4xp150](Llama-3.3-70B-Instruct_p150x4.md)
- [8xp150](Llama-3.3-70B-Instruct_p150x8.md)

#### Back links

- [Tenstorrent Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Llama-3.3-70B-Instruct Inference Server on Tenstorrent Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

The default model weights for this implementation is `Llama-3.3-70B-Instruct`, the following weights are supported as well:

- `Llama-3.1-70B`
- `Llama-3.1-70B-Instruct`
- `DeepSeek-R1-Distill-Llama-70B`

To use these weights simply swap `Llama-3.3-70B-Instruct` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.3-70B-Instruct --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B), [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| Model Status | 🟢 Complete |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [llama3-70b-galaxy](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/demos/llama3_70b_galaxy) |
| tt-metal Commit | `a9b09e0` |
| vLLM Commit | `a186bf4` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-a9b09e0-a186bf4` |
