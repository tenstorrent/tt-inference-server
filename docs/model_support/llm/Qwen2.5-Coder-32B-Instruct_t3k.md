# Qwen2.5-Coder-32B-Instruct Tenstorrent Support on WH LoudBox/QuietBox

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen2.5-Coder-32B-Instruct` is also supported on hardware:

- [WH Galaxy](Qwen2.5-Coder-32B-Instruct_galaxy.md)

## Quickstart - Deploy Qwen2.5-Coder-32B-Instruct Inference Server on WH LoudBox/QuietBox

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
  --volume volume_id_Qwen2.5-Coder-32B-Instruct:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-17a5973-aa4ae1e \
  --model Qwen2.5-Coder-32B-Instruct \
  --tt-device t3k
```

**via run.py command**

```bash
python3 run.py --model Qwen2.5-Coder-32B-Instruct --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/17a5973/models/tt_transformers) |
| tt-metal Commit | `17a5973` |
| vLLM Commit | `aa4ae1e` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-17a5973-aa4ae1e` |
