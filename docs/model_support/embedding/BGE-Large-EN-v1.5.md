# BGE-Large-EN-v1.5 Tenstorrent Support

## Run BGE-Large-EN-v1.5 on n150

[Embedding Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model BGE-Large-EN-v1.5 --device n150 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [tt-vllm-plugin](https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-vllm-plugin/tree/2496be4/tt_vllm_plugin) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.2.0-2496be4518bca0a7a5b497a4cda3cfe7e2f59756` |

## Run BGE-Large-EN-v1.5 on n300

[Embedding Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model BGE-Large-EN-v1.5 --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [tt-vllm-plugin](https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-vllm-plugin/tree/2496be4/tt_vllm_plugin) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.2.0-2496be4518bca0a7a5b497a4cda3cfe7e2f59756` |

## Run BGE-Large-EN-v1.5 on TT-LoudBox

[Embedding Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model BGE-Large-EN-v1.5 --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 4 |
| Implementation Code | [tt-vllm-plugin](https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-vllm-plugin/tree/2496be4/tt_vllm_plugin) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.2.0-2496be4518bca0a7a5b497a4cda3cfe7e2f59756` |

## Run BGE-Large-EN-v1.5 on Tenstorrent Galaxy

[Embedding Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model BGE-Large-EN-v1.5 --device galaxy --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Implementation Code | [tt-vllm-plugin](https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-vllm-plugin/tree/2496be4/tt_vllm_plugin) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.2.0-2496be4518bca0a7a5b497a4cda3cfe7e2f59756` |
