# DeepSeek-R1-0528 Tenstorrent Support on Dual WH Galaxy

#### Useful links

- [Dual WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`DeepSeek-R1-0528` is also supported on hardware:

- [Quad WH Galaxy](DeepSeek-R1-0528_quad_galaxy.md)
- [WH Galaxy](DeepSeek-R1-0528_galaxy.md)

## Quickstart - Deploy DeepSeek-R1-0528 Inference Server on Dual WH Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal/README.md) inference engine.

**Note:** Dual WH Galaxy requires multi-host deployment with Controller and Worker containers. See the [Multi-Host Deployment Guide](../../multihost_deployment.md) for detailed setup instructions.

**via run.py command**

```bash
python3 run.py --model DeepSeek-R1-0528 --device dual_galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 256 |
| Max Context Length | 32768 |
| Implementation Code | [deepseek-r1-galaxy](https://github.com/tenstorrent/tt-metal/tree/805f43d/models/demos/deepseek_v3) |
| tt-metal Commit | `805f43d` |
| vLLM Commit | `a45c614` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-multihost-ubuntu-22.04-amd64:0.12.0-805f43d-a45c614` |
