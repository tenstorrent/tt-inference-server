# DeepSeek-R1-0528 Tenstorrent Support on WH Galaxy

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy DeepSeek-R1-0528 Inference Server on WH Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal-llama3/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model DeepSeek-R1-0528 --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 256 |
| Max Context Length | 65536 |
| Implementation Code | [deepseek-r1-galaxy](https://github.com/tenstorrent/tt-metal/tree/e3d97e5/models/demos/deepseek_v3) |
| tt-metal Commit | `e3d97e5` |
| vLLM Commit | `a186bf4` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e3d97e5-a186bf4` |
