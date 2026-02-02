# Llama-3.3-70B-Instruct Tenstorrent Support

## Run Llama-3.3-70B-Instruct on TT-LoudBox

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.3-70B-Instruct`, the following weights are supported as well:

- `Llama-3.1-70B`
- `Llama-3.1-70B-Instruct`
- `DeepSeek-R1-Distill-Llama-70B`

To use these weights simply swap `Llama-3.3-70B-Instruct` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.3-70B-Instruct --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B), [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/0b10c51/models/tt_transformers) |
| tt-metal Commit | `0b10c51` |
| vLLM Commit | `3499ffa` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-0b10c51-3499ffa` |

## Run Llama-3.3-70B-Instruct on Tenstorrent Galaxy

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.3-70B-Instruct`, the following weights are supported as well:

- `Llama-3.1-70B`
- `Llama-3.1-70B-Instruct`
- `DeepSeek-R1-Distill-Llama-70B`

To use these weights simply swap `Llama-3.3-70B-Instruct` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.3-70B-Instruct --device galaxy --workflow server --docker-server
```

### Model Parameters

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

## Run Llama-3.3-70B-Instruct on Tenstorrent Galaxy (GALAXY_T3K)

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.3-70B-Instruct`, the following weights are supported as well:

- `Llama-3.1-70B`
- `Llama-3.1-70B-Instruct`
- `DeepSeek-R1-Distill-Llama-70B`

To use these weights simply swap `Llama-3.3-70B-Instruct` for your desired weights in commands below.

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
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-v0.62.0-rc33-e7c329b` |

## Run Llama-3.3-70B-Instruct on 4xp150

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.3-70B-Instruct`, the following weights are supported as well:

- `Llama-3.1-70B`
- `Llama-3.1-70B-Instruct`
- `DeepSeek-R1-Distill-Llama-70B`

To use these weights simply swap `Llama-3.3-70B-Instruct` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.3-70B-Instruct --device p150x4 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B), [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/55fd115/models/tt_transformers) |
| tt-metal Commit | `55fd115` |
| vLLM Commit | `aa4ae1e` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-55fd115-aa4ae1e` |

## Run Llama-3.3-70B-Instruct on 8xp150

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.3-70B-Instruct`, the following weights are supported as well:

- `Llama-3.1-70B`
- `Llama-3.1-70B-Instruct`
- `DeepSeek-R1-Distill-Llama-70B`

To use these weights simply swap `Llama-3.3-70B-Instruct` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.3-70B-Instruct --device p150x8 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B), [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/55fd115/models/tt_transformers) |
| tt-metal Commit | `55fd115` |
| vLLM Commit | `aa4ae1e` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-55fd115-aa4ae1e` |
