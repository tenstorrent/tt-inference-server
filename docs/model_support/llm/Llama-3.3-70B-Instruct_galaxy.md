# Llama-3.3-70B-Instruct Tenstorrent Support on WH Galaxy

Supported weights variants for this model implementation are:

- `Llama-3.3-70B-Instruct`: [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) **(default)** 
- `Llama-3.1-70B`: [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)
- `Llama-3.1-70B-Instruct`: [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
- `DeepSeek-R1-Distill-Llama-70B`: [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)

To use non-default weights, replace `Llama-3.3-70B-Instruct` in commands below.

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Llama-3.3-70B-Instruct` is also supported on hardware:

- [BH LoudBox](Llama-3.3-70B-Instruct_p150x8.md)
- [BH QuietBox](Llama-3.3-70B-Instruct_p150x4.md)
- [WH LoudBox/QuietBox](Llama-3.3-70B-Instruct_t3k.md)

## Quickstart - Deploy Llama-3.3-70B-Instruct Inference Server on WH Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

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
| Implementation Code | [llama3-70b-galaxy](https://github.com/tenstorrent/tt-metal/tree/65718bb/models/demos/llama3_70b_galaxy) |
| tt-metal Commit | `65718bb` |
| vLLM Commit | `409b1cd` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-65718bb-409b1cd` |

---

## GALAXY_T3K Configuration

### Quickstart - Deploy on WH Galaxy

**via run.py command**

```bash
python3 run.py --model Llama-3.3-70B-Instruct --device galaxy_t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B), [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc33/models/tt_transformers) |
| tt-metal Commit | `v0.62.0-rc33` |
| vLLM Commit | `e7c329b` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-v0.62.0-rc33-e7c329b` |
