# Llama-3.1-70B Tenstorrent Support on WH LoudBox/QuietBox

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Llama-3.1-70B Inference Server on WH LoudBox/QuietBox

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Llama-3.1-70B --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Max Context Length | 131072 |
| Implementation Code | [forge-vllm-plugin](https://github.com/tenstorrent/tt-xla/tree/main/tree/2496be4/integrations/vllm_plugin) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.11.1-2496be4` |
