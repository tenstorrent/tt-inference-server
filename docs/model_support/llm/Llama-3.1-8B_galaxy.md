# Llama-3.1-8B Tenstorrent Support on WH Galaxy

Supported weights variants for this model implementation are:

- `Llama-3.1-8B`: [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) **(default)** 
- `Llama-3.1-8B-Instruct`: [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

To use non-default weights, replace `Llama-3.1-8B` in commands below.

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Llama-3.1-8B` is also supported on hardware:

- [BH LoudBox](Llama-3.1-8B_p150x8.md)
- [BH 4xP150](Llama-3.1-8B_p150x4.md)
- [P100/P150](Llama-3.1-8B_p100.md)
- [WH LoudBox/QuietBox](Llama-3.1-8B_t3k.md)
- [N150/N300](Llama-3.1-8B_n150.md)

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
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.1-bac8b34-7c6685a \
  --model Llama-3.1-8B \
  --tt-device galaxy
```

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 128 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/bac8b34/models/tt_transformers) |
| tt-metal Commit | `bac8b34` |
| vLLM Commit | `7c6685a` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.1-bac8b34-7c6685a` |

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
  --volume volume_id_Llama-3.1-8B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.1-bac8b34-7c6685a \
  --model Llama-3.1-8B \
  --tt-device galaxy_t3k
```

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device galaxy_t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/bac8b34/models/tt_transformers) |
| tt-metal Commit | `bac8b34` |
| vLLM Commit | `7c6685a` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.1-bac8b34-7c6685a` |
