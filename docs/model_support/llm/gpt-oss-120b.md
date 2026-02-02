# gpt-oss-120b Tenstorrent Support

## Run gpt-oss-120b on TT-LoudBox

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model gpt-oss-120b --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Model Status | 🟡 Functional |
| Max Batch Size | 1 |
| Max Context Length | 1024 |
| Implementation Code | [gpt-oss](https://github.com/tenstorrent/tt-metal/tree/60ffb199/models/demos/gpt_oss) |
| tt-metal Commit | `60ffb199` |
| vLLM Commit | `3499ffa1` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-60ffb199-3499ffa1` |

## Run gpt-oss-120b on Tenstorrent Galaxy

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model gpt-oss-120b --device galaxy --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| Model Status | 🟡 Functional |
| Max Batch Size | 128 |
| Max Context Length | 131072 |
| Implementation Code | [gpt-oss](https://github.com/tenstorrent/tt-metal/tree/60ffb199/models/demos/gpt_oss) |
| tt-metal Commit | `60ffb199` |
| vLLM Commit | `3499ffa1` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-60ffb199-3499ffa1` |
