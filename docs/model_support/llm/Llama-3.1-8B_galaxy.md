# Llama-3.1-8B Tenstorrent Support on WH Galaxy

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Llama-3.1-8B` is also supported on hardware:

- [BH LoudBox](Llama-3.1-8B_p150x8.md)
- [BH P300](Llama-3.1-8B_p300.md)
- [BH QuietBox 2](Llama-3.1-8B_p300x2.md)
- [P150](Llama-3.1-8B_p150.md)

## Quickstart - Deploy Llama-3.1-8B Inference Server on WH Galaxy

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
  --volume volume_id_Llama-3.1-8B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.0-bac8b34-7c6685a \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tt-device galaxy
```

**via run.py command**

```bash
python3 run.py --model meta-llama/Llama-3.1-8B-Instruct --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 128 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/bac8b34/models/tt_transformers) |
| tt-metal Commit | `bac8b34` |
| vLLM Commit | `7c6685a` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.0-bac8b34-7c6685a` |
