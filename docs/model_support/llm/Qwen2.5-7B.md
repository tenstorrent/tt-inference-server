# Qwen2.5-7B Tenstorrent Support

## Run Qwen2.5-7B on n300

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Qwen2.5-7B`, the following weights are supported as well:

- `Qwen2.5-7B-Instruct`

To use these weights simply swap `Qwen2.5-7B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Qwen2.5-7B --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B), [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/5b5db8a/models/tt_transformers) |
| tt-metal Commit | `5b5db8a` |
| vLLM Commit | `e771fff` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-5b5db8a-e771fff` |

## Run Qwen2.5-7B on 4xn150

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Qwen2.5-7B`, the following weights are supported as well:

- `Qwen2.5-7B-Instruct`

To use these weights simply swap `Qwen2.5-7B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Qwen2.5-7B --device n150x4 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B), [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/5b5db8a/models/tt_transformers) |
| tt-metal Commit | `5b5db8a` |
| vLLM Commit | `e771fff` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-5b5db8a-e771fff` |
