# gemma-3-27b-it Tenstorrent Support on WH Galaxy

Supported weights variants for this model implementation are:

- `gemma-3-27b-it`: [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) **(default)** 
- `medgemma-27b-it`: [google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it)

To use non-default weights, replace `gemma-3-27b-it` in commands below.

#### Useful links

- [WH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other vlm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`gemma-3-27b-it` is also supported on hardware:

- [WH LoudBox/QuietBox](gemma-3-27b-it_t3k.md)

## Quickstart - Deploy gemma-3-27b-it Inference Server on WH Galaxy

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
  --volume volume_id_gemma-3-27b-it:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-a8c5af0-1abfcfc \
  --model gemma-3-27b-it \
  --tt-device galaxy
```

**via run.py command**

```bash
python3 run.py --model gemma-3-27b-it --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it), [google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a8c5af0/models/tt_transformers) |
| tt-metal Commit | `a8c5af0` |
| vLLM Commit | `1abfcfc` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-a8c5af0-1abfcfc` |

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
  --volume volume_id_gemma-3-27b-it:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-a8c5af0-1abfcfc \
  --model gemma-3-27b-it \
  --tt-device galaxy_t3k
```

**via run.py command**

```bash
python3 run.py --model gemma-3-27b-it --device galaxy_t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it), [google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a8c5af0/models/tt_transformers) |
| tt-metal Commit | `a8c5af0` |
| vLLM Commit | `1abfcfc` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-a8c5af0-1abfcfc` |
