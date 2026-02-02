# AFM-4.5B Tenstorrent Support

## Run AFM-4.5B on n300

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model AFM-4.5B --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [arcee-ai/AFM-4.5B](https://huggingface.co/arcee-ai/AFM-4.5B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 65536 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/ae65ee5/models/tt_transformers) |
| tt-metal Commit | `ae65ee5` |
| vLLM Commit | `35f023f` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-ae65ee5-35f023f` |

## Run AFM-4.5B on TT-LoudBox

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model AFM-4.5B --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [arcee-ai/AFM-4.5B](https://huggingface.co/arcee-ai/AFM-4.5B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 65536 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/ae65ee5/models/tt_transformers) |
| tt-metal Commit | `ae65ee5` |
| vLLM Commit | `35f023f` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-ae65ee5-35f023f` |
