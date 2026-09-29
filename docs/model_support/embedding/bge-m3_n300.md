# bge-m3 Tenstorrent Support on N300

#### Useful links

- [N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other embedding models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`bge-m3` is also supported on hardware:

- [BH QuietBox 2](bge-m3_p300x2.md)

## Quickstart - Deploy bge-m3 Inference Server on n300

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
  --volume volume_id_bge-m3:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-media-inference-server:0.20.0-bc294789ec3 \
  --model bge-m3 \
  --tt-device n300
```

**via run.py command**

```bash
python3 run.py --model bge-m3 --device n300 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 12 |
| Implementation Code | [tt-vllm-plugin](https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-vllm-plugin/tree/bc294789ec3/tt_vllm_plugin) |
| tt-metal Commit | `bc294789ec3` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.20.0-bc294789ec3` |
