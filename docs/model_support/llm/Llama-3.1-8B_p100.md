# Llama-3.1-8B Tenstorrent Support on P100

Supported weights variants for this model implementation are:

- `Llama-3.1-8B`: [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) **(default)** 
- `Llama-3.1-8B-Instruct`: [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

> **Note:** The default `meta-llama/Llama-3.1-8B` is a **base** (pretrained, not instruction-tuned) model. For conversational/chat use cases, substitute `meta-llama/Llama-3.1-8B-Instruct` — it follows instructions and produces appropriate chat responses out of the box. The base model continues text and is not suited to chat without fine-tuning.

To use non-default weights, replace `meta-llama/Llama-3.1-8B` in commands below.

#### Useful links

- [P100 details](https://tenstorrent.com/hardware/blackhole)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Llama-3.1-8B` is also supported on hardware:

- [WH Galaxy](Llama-3.1-8B_galaxy.md)
- [BH LoudBox](Llama-3.1-8B_p150x8.md)
- [BH 4xP150](Llama-3.1-8B_p150x4.md)
- [BH P300](Llama-3.1-8B_p300.md)
- [BH QuietBox 2](Llama-3.1-8B_p300x2.md)
- [P150](Llama-3.1-8B_p150.md)
- [WH LoudBox/QuietBox](Llama-3.1-8B_t3k.md)
- [N150](Llama-3.1-8B_n150.md)
- [N300](Llama-3.1-8B_n300.md)

## Quickstart - Deploy Llama-3.1-8B Inference Server on p100

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [vLLM (tt-metal integration fork)](../../../vllm-tt-metal/README.md) inference engine.

**docker run command**

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "TRANSFORMERS_OFFLINE=1" \
  --env "HF_HUB_OFFLINE=1" \
  --env "HF_DATASETS_OFFLINE=1" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_Llama-3.1-8B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.18.0-c49bb76-6b4a3a7 \
  --model meta-llama/Llama-3.1-8B \
  --tt-device p100
```

The inference container does not have outbound internet access. The `TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE`, and `HF_DATASETS_OFFLINE` flags prevent Transformers and vLLM from attempting to reach `huggingface.co` at runtime; ensure model weights are pre-downloaded to the mounted cache/volume before starting the container.

**via run.py command**

```bash
python3 run.py --model meta-llama/Llama-3.1-8B --device p100 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 65536 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c49bb76/models/tt_transformers) |
| tt-metal Commit | `c49bb76` |
| vLLM Commit | `6b4a3a7` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.18.0-c49bb76-6b4a3a7` |
