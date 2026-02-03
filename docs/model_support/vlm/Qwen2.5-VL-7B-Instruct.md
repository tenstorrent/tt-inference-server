# Qwen2.5-VL-7B-Instruct Tenstorrent Support

## Run Qwen2.5-VL-7B-Instruct on n150

[VLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model Qwen2.5-VL-7B-Instruct --device n150 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c18569e/models/tt_transformers) |
| tt-metal Commit | `c18569e` |
| vLLM Commit | `b2894d3` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-c18569e-b2894d3` |

## Run Qwen2.5-VL-7B-Instruct on n300

[VLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model Qwen2.5-VL-7B-Instruct --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c18569e/models/tt_transformers) |
| tt-metal Commit | `c18569e` |
| vLLM Commit | `b2894d3` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-c18569e-b2894d3` |

## Run Qwen2.5-VL-7B-Instruct on TT-LoudBox

[VLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model Qwen2.5-VL-7B-Instruct --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c18569e/models/tt_transformers) |
| tt-metal Commit | `c18569e` |
| vLLM Commit | `b2894d3` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-c18569e-b2894d3` |
