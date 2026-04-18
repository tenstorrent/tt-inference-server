# AFM-4.5B Tenstorrent Support on N150/N300

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`AFM-4.5B` is also supported on hardware:

- [WH LoudBox/QuietBox](AFM-4.5B_t3k.md)

## Quickstart - Deploy AFM-4.5B Inference Server on n300

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
  --volume volume_id_AFM-4.5B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.3.0-ae65ee5-35f023f \
  --model AFM-4.5B \
  --tt-device n300
```

**via run.py command**

```bash
python3 run.py --model AFM-4.5B --device n300 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [arcee-ai/AFM-4.5B](https://huggingface.co/arcee-ai/AFM-4.5B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 65536 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/ae65ee5/models/tt_transformers) |
| tt-metal Commit | `ae65ee5` |
| vLLM Commit | `35f023f` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.3.0-ae65ee5-35f023f` |
