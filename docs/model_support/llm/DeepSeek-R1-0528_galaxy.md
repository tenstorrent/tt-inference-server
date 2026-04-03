# DeepSeek-R1-0528 Tenstorrent Support on WH Galaxy

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`DeepSeek-R1-0528` is also supported on hardware:

- [Dual WH Galaxy](DeepSeek-R1-0528_dual_galaxy.md)
- [Quad WH Galaxy](DeepSeek-R1-0528_quad_galaxy.md)

## Quickstart - Deploy DeepSeek-R1-0528 Inference Server on WH Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal/README.md) inference engine.

**docker run command**

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_DeepSeek-R1-0528:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.12.0-805f43d-a45c614 \
  --model DeepSeek-R1-0528 \
  --tt-device galaxy
```

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
| Max Context Length | 2048 |
| Implementation Code | [deepseek-r1-galaxy](https://github.com/tenstorrent/tt-metal/tree/805f43d/models/demos/deepseek_v3) |
| tt-metal Commit | `805f43d` |
| vLLM Commit | `a45c614` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.12.0-805f43d-a45c614` |
