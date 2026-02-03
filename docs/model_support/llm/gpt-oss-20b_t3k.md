# gpt-oss-20b Tenstorrent Support on TT-LoudBox

gpt-oss-20b is also supported on:

- [Tenstorrent Galaxy](gpt-oss-20b_galaxy.md)
- [Tenstorrent Galaxy (GALAXY_T3K)](gpt-oss-20b_galaxy_t3k.md)

#### Back links

- [TT-LoudBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy gpt-oss-20b Inference Server on TT-LoudBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model gpt-oss-20b --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Model Status | 🟡 Functional |
| Max Batch Size | 1 |
| Max Context Length | 1024 |
| Implementation Code | [gpt-oss](https://github.com/tenstorrent/tt-metal/tree/60ffb199/models/demos/gpt_oss) |
| tt-metal Commit | `60ffb199` |
| vLLM Commit | `3499ffa1` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-60ffb199-3499ffa1` |
