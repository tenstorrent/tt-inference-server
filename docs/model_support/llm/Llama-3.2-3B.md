# Llama-3.2-3B Tenstorrent Support

## Run Llama-3.2-3B on n150

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.2-3B`, the following weights are supported as well:

- `Llama-3.2-3B-Instruct`

To use these weights simply swap `Llama-3.2-3B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.2-3B --device n150 --workflow server --docker-server
```

### Model Parameters

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

## Run Llama-3.2-3B on n300

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.2-3B`, the following weights are supported as well:

- `Llama-3.2-3B-Instruct`

To use these weights simply swap `Llama-3.2-3B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.2-3B --device n300 --workflow server --docker-server
```

### Model Parameters

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

## Run Llama-3.2-3B on TT-LoudBox

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.2-3B`, the following weights are supported as well:

- `Llama-3.2-3B-Instruct`

To use these weights simply swap `Llama-3.2-3B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.2-3B --device t3k --workflow server --docker-server
```

### Model Parameters

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
