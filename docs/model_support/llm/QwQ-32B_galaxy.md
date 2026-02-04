# QwQ-32B Tenstorrent Support on WH Galaxy

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`QwQ-32B` is also supported on hardware:

- [WH LoudBox/QuietBox](QwQ-32B_t3k.md)

## Quickstart - Deploy QwQ-32B Inference Server on WH Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model QwQ-32B --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 128 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/e95ffa5/models/tt_transformers) |
| tt-metal Commit | `e95ffa5` |
| vLLM Commit | `48eba14` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e95ffa5-48eba14` |

---

## GALAXY_T3K Configuration

### Quickstart - Deploy on WH Galaxy

**via run.py command**

```bash
python3 run.py --model QwQ-32B --device galaxy_t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/e95ffa5/models/tt_transformers) |
| tt-metal Commit | `e95ffa5` |
| vLLM Commit | `48eba14` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e95ffa5-48eba14` |
