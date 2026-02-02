# gemma-3-27b-it Tenstorrent Support

## Run gemma-3-27b-it on TT-LoudBox

[VLM Model Support Table](../vlm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `gemma-3-27b-it`, the following weights are supported as well:

- `medgemma-27b-it`

To use these weights simply swap `gemma-3-27b-it` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model gemma-3-27b-it --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it), [google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/0b10c51/models/tt_transformers) |
| tt-metal Commit | `0b10c51` |
| vLLM Commit | `3499ffa` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-0b10c51-3499ffa` |

## Run gemma-3-27b-it on Tenstorrent Galaxy

[VLM Model Support Table](../vlm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `gemma-3-27b-it`, the following weights are supported as well:

- `medgemma-27b-it`

To use these weights simply swap `gemma-3-27b-it` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model gemma-3-27b-it --device galaxy --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it), [google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/0b10c51/models/tt_transformers) |
| tt-metal Commit | `0b10c51` |
| vLLM Commit | `3499ffa` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-0b10c51-3499ffa` |

## Run gemma-3-27b-it on Tenstorrent Galaxy (GALAXY_T3K)

[VLM Model Support Table](../vlm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `gemma-3-27b-it`, the following weights are supported as well:

- `medgemma-27b-it`

To use these weights simply swap `gemma-3-27b-it` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model gemma-3-27b-it --device galaxy_t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it), [google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/0b10c51/models/tt_transformers) |
| tt-metal Commit | `0b10c51` |
| vLLM Commit | `3499ffa` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-0b10c51-3499ffa` |
