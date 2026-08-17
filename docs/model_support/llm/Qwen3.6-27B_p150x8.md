# Qwen3.6-27B Tenstorrent Support on BH LoudBox

#### Useful links

- [BH LoudBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen3.6-27B` is also supported on hardware:

- [BH QuietBox 2](Qwen3.6-27B_p300x2.md)

## Quickstart - Deploy Qwen3.6-27B Inference Server on BH LoudBox

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
  --volume volume_id_Qwen3.6-27B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af \
  --model Qwen3.6-27B \
  --tt-device p150x8
```

**via run.py command**

```bash
python3 run.py --model Qwen3.6-27B --device p150x8 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 262144 |
| Implementation Code | [qwen36-blackhole](https://github.com/tenstorrent/tt-metal/tree/de59f8a/models/demos/blackhole/qwen36) |
| tt-metal Commit | `de59f8a` |
| vLLM Commit | `03fa3af` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.20.0-de59f8a-03fa3af` |
