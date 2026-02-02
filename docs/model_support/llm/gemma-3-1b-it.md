# gemma-3-1b-it Tenstorrent Support

## Run gemma-3-1b-it on n150

[LLM Model Support Table](../llm_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model gemma-3-1b-it --device n150 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 32768 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c254ee3/models/tt_transformers) |
| tt-metal Commit | `c254ee3` |
| vLLM Commit | `c4f2327` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-c254ee3-c4f2327` |
