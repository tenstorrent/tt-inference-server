# Qwen3-8B Tenstorrent Support on WH Galaxy

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen3-8B` is also supported on hardware:

- [WH LoudBox/QuietBox](Qwen3-8B_t3k.md)
- [N150/N300](Qwen3-8B_n150.md)

## Quickstart - Deploy Qwen3-8B Inference Server on WH Galaxy

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
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-a8c5af0-1abfcfc \
  --model Qwen3-8B \
  --tt-device galaxy
```

**via run.py command**

```bash
python3 run.py --model Qwen3-8B --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 128 |
| Max Context Length | 40960 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a8c5af0/models/tt_transformers) |
| tt-metal Commit | `a8c5af0` |
| vLLM Commit | `1abfcfc` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-a8c5af0-1abfcfc` |

---

## GALAXY_T3K Configuration

### Quickstart - Deploy on WH Galaxy

**docker run command**

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_Qwen3-8B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-a8c5af0-1abfcfc \
  --model Qwen3-8B \
  --tt-device galaxy_t3k
```

**via run.py command**

```bash
python3 run.py --model Qwen3-8B --device galaxy_t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 40960 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a8c5af0/models/tt_transformers) |
| tt-metal Commit | `a8c5af0` |
| vLLM Commit | `1abfcfc` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-a8c5af0-1abfcfc` |
