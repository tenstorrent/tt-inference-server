# Qwen3-8B Tenstorrent Support

## Run Qwen3-8B on n150

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model Qwen3-8B --device n150 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 40960 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/e95ffa5/models/tt_transformers) |
| tt-metal Commit | `e95ffa5` |
| vLLM Commit | `48eba14` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e95ffa5-48eba14` |

## Run Qwen3-8B on n300

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model Qwen3-8B --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 40960 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/e95ffa5/models/tt_transformers) |
| tt-metal Commit | `e95ffa5` |
| vLLM Commit | `48eba14` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e95ffa5-48eba14` |

## Run Qwen3-8B on TT-LoudBox

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model Qwen3-8B --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 40960 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/e95ffa5/models/tt_transformers) |
| tt-metal Commit | `e95ffa5` |
| vLLM Commit | `48eba14` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e95ffa5-48eba14` |

## Run Qwen3-8B on Tenstorrent Galaxy

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model Qwen3-8B --device galaxy --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 128 |
| Max Context Length | 40960 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/e95ffa5/models/tt_transformers) |
| tt-metal Commit | `e95ffa5` |
| vLLM Commit | `48eba14` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e95ffa5-48eba14` |

## Run Qwen3-8B on Tenstorrent Galaxy (GALAXY_T3K)

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model Qwen3-8B --device galaxy_t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 40960 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/e95ffa5/models/tt_transformers) |
| tt-metal Commit | `e95ffa5` |
| vLLM Commit | `48eba14` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e95ffa5-48eba14` |
