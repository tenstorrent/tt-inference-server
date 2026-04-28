# Llama-3.2-3B Tenstorrent Support on N150/N300

Supported weights variants for this model implementation are:

- `Llama-3.2-3B`: [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) **(default)** 
- `Llama-3.2-3B-Instruct`: [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

To use non-default weights, replace `Llama-3.2-3B` in commands below.

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Llama-3.2-3B` is also supported on hardware:

- [WH LoudBox/QuietBox](Llama-3.2-3B_t3k.md)

## Quickstart - Deploy Llama-3.2-3B Inference Server on n150

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
  --volume volume_id_Llama-3.2-3B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.3.0-a8c5af0-1abfcfc \
  --model Llama-3.2-3B \
  --tt-device n150
```

**via run.py command**

```bash
python3 run.py --model Llama-3.2-3B --device n150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B), [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a8c5af0/models/tt_transformers) |
| tt-metal Commit | `a8c5af0` |
| vLLM Commit | `1abfcfc` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.3.0-a8c5af0-1abfcfc` |

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
  --volume volume_id_Llama-3.2-3B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.3.0-a8c5af0-1abfcfc \
  --model Llama-3.2-3B \
  --tt-device n300
```

**via run.py command**

```bash
python3 run.py --model Llama-3.2-3B --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B), [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a8c5af0/models/tt_transformers) |
| tt-metal Commit | `a8c5af0` |
| vLLM Commit | `1abfcfc` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.3.0-a8c5af0-1abfcfc` |
