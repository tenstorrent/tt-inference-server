# gemma-3-4b-it Tenstorrent Support on N150/N300

Supported weights variants for this model implementation are:

- `gemma-3-4b-it`: [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) **(default)** 
- `medgemma-4b-it`: [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)

To use non-default weights, replace `gemma-3-4b-it` in commands below.

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other vlm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy gemma-3-4b-it Inference Server on n150

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
  --volume volume_id_gemma-3-4b-it:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-aecd1d7-0da90eb \
  --model gemma-3-4b-it \
  --tt-device n150
```

**via run.py command**

```bash
python3 run.py --model gemma-3-4b-it --device n150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it), [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/aecd1d7/models/tt_transformers) |
| tt-metal Commit | `aecd1d7` |
| vLLM Commit | `0da90eb` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-aecd1d7-0da90eb` |

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
  --volume volume_id_gemma-3-4b-it:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-aecd1d7-0da90eb \
  --model gemma-3-4b-it \
  --tt-device n300
```

**via run.py command**

```bash
python3 run.py --model gemma-3-4b-it --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it), [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/aecd1d7/models/tt_transformers) |
| tt-metal Commit | `aecd1d7` |
| vLLM Commit | `0da90eb` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-aecd1d7-0da90eb` |
