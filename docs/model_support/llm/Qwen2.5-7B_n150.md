# Qwen2.5-7B Tenstorrent Support on N150/N300

Supported weights variants for this model implementation are:

- `Qwen2.5-7B`: [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) **(default)** 
- `Qwen2.5-7B-Instruct`: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

To use non-default weights, replace `Qwen2.5-7B` in commands below.

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Qwen2.5-7B Inference Server on n300

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
  --volume volume_id_Qwen2.5-7B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-5b5db8a-e771fff \
  --model Qwen2.5-7B \
  --tt-device n300
```

**via run.py command**

```bash
python3 run.py --model Qwen2.5-7B --device n300 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B), [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/5b5db8a/models/tt_transformers) |
| tt-metal Commit | `5b5db8a` |
| vLLM Commit | `e771fff` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-5b5db8a-e771fff` |
