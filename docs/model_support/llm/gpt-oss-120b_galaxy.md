# gpt-oss-120b Tenstorrent Support on WH Galaxy

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`gpt-oss-120b` is also supported on:

- [WH LoudBox/QuietBox](gpt-oss-120b_t3k.md)

## Quickstart - Deploy gpt-oss-120b Inference Server on WH Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model gpt-oss-120b --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

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
