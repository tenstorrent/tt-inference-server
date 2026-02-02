# Qwen3-Embedding-4B Tenstorrent Support

## Run Qwen3-Embedding-4B on n150

[Embedding Model Support Table](../embedding_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model Qwen3-Embedding-4B --device n150 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [forge-vllm-plugin](https://github.com/tenstorrent/tt-xla/tree/main/tree/2496be4/integrations/vllm_plugin) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:a9b09e0b611da6deb4d8972e8296148fd864e5fd_98dcf62_60920940673` |

## Run Qwen3-Embedding-4B on n300

[Embedding Model Support Table](../embedding_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model Qwen3-Embedding-4B --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [forge-vllm-plugin](https://github.com/tenstorrent/tt-xla/tree/main/tree/2496be4/integrations/vllm_plugin) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:a9b09e0b611da6deb4d8972e8296148fd864e5fd_98dcf62_60920940673` |

## Run Qwen3-Embedding-4B on TT-LoudBox

[Embedding Model Support Table](../embedding_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model Qwen3-Embedding-4B --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 4 |
| Implementation Code | [forge-vllm-plugin](https://github.com/tenstorrent/tt-xla/tree/main/tree/2496be4/integrations/vllm_plugin) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:a9b09e0b611da6deb4d8972e8296148fd864e5fd_98dcf62_60920940673` |

## Run Qwen3-Embedding-4B on Tenstorrent Galaxy

[Embedding Model Support Table](../embedding_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model Qwen3-Embedding-4B --device galaxy --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Implementation Code | [forge-vllm-plugin](https://github.com/tenstorrent/tt-xla/tree/main/tree/2496be4/integrations/vllm_plugin) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:a9b09e0b611da6deb4d8972e8296148fd864e5fd_98dcf62_60920940673` |
