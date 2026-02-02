# Llama-3.2-11B-Vision Tenstorrent Support

## Run Llama-3.2-11B-Vision on n300

[VLM Model Support Table](../vlm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.2-11B-Vision`, the following weights are supported as well:

- `Llama-3.2-11B-Vision-Instruct`

To use these weights simply swap `Llama-3.2-11B-Vision` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.2-11B-Vision --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision), [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 16 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/v0.61.1-rc1/models/tt_transformers) |
| tt-metal Commit | `v0.61.1-rc1` |
| vLLM Commit | `5cbc982` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-v0.61.1-rc1-5cbc982` |

## Run Llama-3.2-11B-Vision on TT-LoudBox

[VLM Model Support Table](../vlm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.2-11B-Vision`, the following weights are supported as well:

- `Llama-3.2-11B-Vision-Instruct`

To use these weights simply swap `Llama-3.2-11B-Vision` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.2-11B-Vision --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision), [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 16 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/v0.61.1-rc1/models/tt_transformers) |
| tt-metal Commit | `v0.61.1-rc1` |
| vLLM Commit | `5cbc982` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-v0.61.1-rc1-5cbc982` |
