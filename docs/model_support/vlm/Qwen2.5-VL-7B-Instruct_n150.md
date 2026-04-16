# Qwen2.5-VL-7B-Instruct Tenstorrent Support on N150/N300

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other vlm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Qwen2.5-VL-7B-Instruct Inference Server on n150

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
  --volume volume_id_Qwen2.5-VL-7B-Instruct:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-c18569e-b2894d3 \
  --model Qwen2.5-VL-7B-Instruct \
  --tt-device n150
```

**via run.py command**

```bash
python3 run.py --model Qwen2.5-VL-7B-Instruct --device n150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 32768 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c18569e/models/tt_transformers) |
| tt-metal Commit | `c18569e` |
| vLLM Commit | `b2894d3` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-c18569e-b2894d3` |

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
  --volume volume_id_Qwen2.5-VL-7B-Instruct:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-c18569e-b2894d3 \
  --model Qwen2.5-VL-7B-Instruct \
  --tt-device n300
```

**via run.py command**

```bash
python3 run.py --model Qwen2.5-VL-7B-Instruct --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 32768 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c18569e/models/tt_transformers) |
| tt-metal Commit | `c18569e` |
| vLLM Commit | `b2894d3` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-c18569e-b2894d3` |
