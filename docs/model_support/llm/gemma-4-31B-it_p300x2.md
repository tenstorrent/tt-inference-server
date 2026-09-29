# gemma-4-31B-it Tenstorrent Support on BH QuietBox 2

#### Useful links

- [BH QuietBox 2 details](https://tenstorrent.com/hardware/tt-quietbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy gemma-4-31B-it Inference Server on BH QuietBox 2

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
  --volume volume_id_gemma-4-31B-it:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.18.0-c49bb76-6b4a3a7 \
  --model google/gemma-4-31B-it \
  --tt-device p300x2
```

**via run.py command**

```bash
python3 run.py --model google/gemma-4-31B-it --device p300x2 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Max Context Length | 49152 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c49bb76/models/tt_transformers) |
| tt-metal Commit | `c49bb76` |
| vLLM Commit | `6b4a3a7` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.18.0-c49bb76-6b4a3a7` |
