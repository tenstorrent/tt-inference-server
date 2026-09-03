# diffusiongemma-26B-A4B-it Tenstorrent Support on BH QuietBox 2

#### Useful links

- [BH QuietBox 2 details](https://tenstorrent.com/hardware/tt-quietbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy diffusiongemma-26B-A4B-it Inference Server on BH QuietBox 2

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
  --volume volume_id_diffusiongemma-26B-A4B-it:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.22.0-bc4c4df-be7d805 \
  --model diffusiongemma-26B-A4B-it \
  --tt-device p300x2
```

**via run.py command**

```bash
python3 run.py --model diffusiongemma-26B-A4B-it --device p300x2 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Max Context Length | 262144 |
| Implementation Code | [diffusion-gemma](https://github.com/tenstorrent/tt-metal/tree/bc4c4df/models/experimental/diffusion_gemma) |
| tt-metal Commit | `bc4c4df` |
| vLLM Commit | `be7d805` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.22.0-bc4c4df-be7d805` |
