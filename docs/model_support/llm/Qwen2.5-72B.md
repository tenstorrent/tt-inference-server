# Qwen2.5-72B Tenstorrent Support

## Run Qwen2.5-72B on TT-LoudBox

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Qwen2.5-72B`, the following weights are supported as well:

- `Qwen2.5-72B-Instruct`

To use these weights simply swap `Qwen2.5-72B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Qwen2.5-72B --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B), [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/13f44c5/models/tt_transformers) |
| tt-metal Commit | `13f44c5` |
| vLLM Commit | `0edd242` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-13f44c5-0edd242` |

## Run Qwen2.5-72B on Tenstorrent Galaxy

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Qwen2.5-72B`, the following weights are supported as well:

- `Qwen2.5-72B-Instruct`

To use these weights simply swap `Qwen2.5-72B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Qwen2.5-72B --device galaxy --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B), [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 128 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/13f44c5/models/tt_transformers) |
| tt-metal Commit | `13f44c5` |
| vLLM Commit | `0edd242` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-13f44c5-0edd242` |

## Run Qwen2.5-72B on Tenstorrent Galaxy (GALAXY_T3K)

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Qwen2.5-72B`, the following weights are supported as well:

- `Qwen2.5-72B-Instruct`

To use these weights simply swap `Qwen2.5-72B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Qwen2.5-72B --device galaxy_t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B), [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/13f44c5/models/tt_transformers) |
| tt-metal Commit | `13f44c5` |
| vLLM Commit | `0edd242` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-13f44c5-0edd242` |
