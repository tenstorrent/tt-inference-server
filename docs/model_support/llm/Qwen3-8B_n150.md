# Qwen3-8B Tenstorrent Support on N150/N300

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen3-8B` is also supported on hardware:

- [WH Galaxy](Qwen3-8B_galaxy.md)
- [WH LoudBox/QuietBox](Qwen3-8B_t3k.md)

## Quickstart - Deploy Qwen3-8B Inference Server on n150

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
  --volume volume_id_Qwen3-8B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-e0e0500-409b1cd \
  --model Qwen3-8B \
  --tt-device n150
```

**via run.py command**

```bash
python3 run.py --model Qwen3-8B --device n150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 40960 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/e0e0500/models/tt_transformers) |
| tt-metal Commit | `e0e0500` |
| vLLM Commit | `409b1cd` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-e0e0500-409b1cd` |

---

## N300 Configuration

### Quickstart - Deploy on n300

**docker run command**

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_Qwen3-8B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-e0e0500-409b1cd \
  --model Qwen3-8B \
  --tt-device n300
```

**via run.py command**

```bash
python3 run.py --model Qwen3-8B --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 40960 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/e0e0500/models/tt_transformers) |
| tt-metal Commit | `e0e0500` |
| vLLM Commit | `409b1cd` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-e0e0500-409b1cd` |
