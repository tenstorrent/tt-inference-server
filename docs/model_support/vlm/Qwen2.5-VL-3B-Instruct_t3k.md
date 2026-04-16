# Qwen2.5-VL-3B-Instruct Tenstorrent Support on WH LoudBox/QuietBox

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other vlm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen2.5-VL-3B-Instruct` is also supported on hardware:

- [N150/N300](Qwen2.5-VL-3B-Instruct_n150.md)

## Quickstart - Deploy Qwen2.5-VL-3B-Instruct Inference Server on WH LoudBox/QuietBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Qwen2.5-VL-3B-Instruct --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c18569e/models/tt_transformers) |
| tt-metal Commit | `c18569e` |
| vLLM Commit | `b2894d3` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-c18569e-b2894d3` |
