# Llama-3.3-70B-Instruct Tenstorrent Support on WH Galaxy

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Llama-3.3-70B-Instruct` is also supported on hardware:

- [BH LoudBox](Llama-3.3-70B-Instruct_p150x8.md)
- [BH QuietBox 2](Llama-3.3-70B-Instruct_p300x2.md)

## Quickstart - Deploy Llama-3.3-70B-Instruct Inference Server on WH Galaxy

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
  --volume volume_id_Llama-3.3-70B-Instruct:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-65718bb-409b1cd \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tt-device galaxy
```

**via run.py command**

```bash
python3 run.py --model meta-llama/Llama-3.3-70B-Instruct --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| Model Status | 🟢 Complete |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [llama3-70b-galaxy](https://github.com/tenstorrent/tt-metal/tree/65718bb/models/demos/llama3_70b_galaxy) |
| tt-metal Commit | `65718bb` |
| vLLM Commit | `409b1cd` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-65718bb-409b1cd` |

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
  --volume volume_id_Llama-3.3-70B-Instruct:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.2.0-v0.62.0-rc33-e7c329b \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tt-device galaxy_t3k
```

**via run.py command**

```bash
python3 run.py --model meta-llama/Llama-3.3-70B-Instruct --device galaxy_t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B), [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/v0.62.0-rc33/models/tt_transformers) |
| tt-metal Commit | `v0.62.0-rc33` |
| vLLM Commit | `e7c329b` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.2.0-v0.62.0-rc33-e7c329b` |
