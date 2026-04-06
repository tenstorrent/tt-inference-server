# Llama-3.3-70B-Instruct Tenstorrent Support on WH LoudBox/QuietBox

Supported weights variants for this model implementation are:

- `Llama-3.3-70B-Instruct`: [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) **(default)** 
- `Llama-3.1-70B`: [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)
- `Llama-3.1-70B-Instruct`: [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
- `DeepSeek-R1-Distill-Llama-70B`: [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)

To use non-default weights, replace `Llama-3.3-70B-Instruct` in commands below.

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Llama-3.3-70B-Instruct` is also supported on hardware:

- [WH Galaxy](Llama-3.3-70B-Instruct_galaxy.md)
- [BH LoudBox](Llama-3.3-70B-Instruct_p150x8.md)
- [BH 4xP150](Llama-3.3-70B-Instruct_p150x4.md)

## Quickstart - Deploy Llama-3.3-70B-Instruct Inference Server on WH LoudBox/QuietBox

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
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.11.1-750ca54-38dee8c \
  --model Llama-3.3-70B-Instruct \
  --tt-device t3k
```

**via run.py command**

```bash
python3 run.py --model Llama-3.3-70B-Instruct --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B), [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/750ca54/models/tt_transformers) |
| tt-metal Commit | `750ca54` |
| vLLM Commit | `38dee8c` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.12.0-750ca54-38dee8c` |
