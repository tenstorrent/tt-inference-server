# Llama-3.2-11B-Vision Tenstorrent Support on N150/N300

The default model weights for this implementation is `Llama-3.2-11B-Vision` ([meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)), the following weights are supported as well:

- `Llama-3.2-11B-Vision-Instruct`: [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

To use these weights simply swap `Llama-3.2-11B-Vision` for your desired weights in commands below.

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other vlm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Llama-3.2-11B-Vision` is also supported on:

- [WH LoudBox/QuietBox](Llama-3.2-11B-Vision_t3k.md)

## Quickstart - Deploy Llama-3.2-11B-Vision Inference Server on n300

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Llama-3.2-11B-Vision --device n300 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

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
