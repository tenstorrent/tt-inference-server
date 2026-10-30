# Kolibri-1-BF16 Tenstorrent Support on BH QuietBox 2

#### Useful links

- [BH QuietBox 2 details](https://tenstorrent.com/hardware/tt-quietbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Kolibri-1-BF16 Inference Server on BH QuietBox 2

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
  --volume volume_id_Kolibri-1-BF16:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.26.0-09acccb-6098002 \
  --model Aleph-Alpha/Kolibri-1-BF16 \
  --tt-device p300x2
```

**via run.py command**

```bash
python3 run.py --model Aleph-Alpha/Kolibri-1-BF16 --device p300x2 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Aleph-Alpha/Kolibri-1-BF16](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 262144 |
| Implementation Code | [kolibri-1-qb2](https://github.com/tenstorrent/tt-metal/tree/09acccb/models/autoports/aleph_alpha_kolibri_1_bf16) |
| tt-metal Commit | `09acccb` |
| vLLM Commit | `6098002` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.26.0-09acccb-6098002` |
