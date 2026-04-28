# Qwen3-32B Tenstorrent Support on WH Galaxy

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen3-32B` is also supported on hardware:

- [BH LoudBox](Qwen3-32B_p150x8.md)
- [WH LoudBox/QuietBox](Qwen3-32B_t3k.md)

## Quickstart - Deploy Qwen3-32B Inference Server on WH Galaxy

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
  --volume volume_id_Qwen3-32B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.1-a8c5af0-1abfcfc \
  --model Qwen3-32B \
  --tt-device galaxy
```

**via run.py command**

```bash
python3 run.py --model Qwen3-32B --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [qwen3-32b-galaxy](https://github.com/tenstorrent/tt-metal/tree/a8c5af0/models/demos/llama3_70b_galaxy) |
| tt-metal Commit | `a8c5af0` |
| vLLM Commit | `1abfcfc` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.1-a8c5af0-1abfcfc` |

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
  --volume volume_id_Qwen3-32B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-e95ffa5-48eba14 \
  --model Qwen3-32B \
  --tt-device galaxy_t3k
```

**via run.py command**

```bash
python3 run.py --model Qwen3-32B --device galaxy_t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/e95ffa5/models/tt_transformers) |
| tt-metal Commit | `e95ffa5` |
| vLLM Commit | `48eba14` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-e95ffa5-48eba14` |
