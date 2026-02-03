# Qwen3-32B Tenstorrent Support on Tenstorrent Galaxy

Qwen3-32B is also supported on:

- [TT-LoudBox](Qwen3-32B_t3k.md)
- [Tenstorrent Galaxy (GALAXY_T3K)](Qwen3-32B_galaxy_t3k.md)
- [8xp150](Qwen3-32B_p150x8.md)

#### Back links

- [Tenstorrent Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Qwen3-32B Inference Server on Tenstorrent Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Qwen3-32B --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |
| Model Status | 🟢 Complete |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [qwen3-32b-galaxy](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/demos/llama3_70b_galaxy) |
| tt-metal Commit | `a9b09e0` |
| vLLM Commit | `a186bf4` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-a9b09e0-a186bf4` |
