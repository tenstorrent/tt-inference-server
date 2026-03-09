# Llama-3.2-1B Tenstorrent Support on N150/N300

Supported weights variants for this model implementation are:

- `Llama-3.2-1B`: [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) **(default)** 
- `Llama-3.2-1B-Instruct`: [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

To use non-default weights, replace `Llama-3.2-1B` in commands below.

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Llama-3.2-1B` is also supported on hardware:

- [WH LoudBox/QuietBox](Llama-3.2-1B_t3k.md)

## Quickstart - Deploy Llama-3.2-1B Inference Server on n150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Llama-3.2-1B --device n150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B), [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/84b4c53/models/tt_transformers) |
| tt-metal Commit | `84b4c53` |
| vLLM Commit | `222ee06` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-84b4c53-222ee06` |

---

## N300 Configuration

### Quickstart - Deploy on n300

**via run.py command**

```bash
python3 run.py --model Llama-3.2-1B --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B), [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/84b4c53/models/tt_transformers) |
| tt-metal Commit | `84b4c53` |
| vLLM Commit | `222ee06` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-84b4c53-222ee06` |
