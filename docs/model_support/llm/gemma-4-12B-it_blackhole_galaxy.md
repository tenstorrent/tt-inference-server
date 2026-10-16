# gemma-4-12B-it Tenstorrent Support on BH Galaxy

#### Useful links

- [BH Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`gemma-4-12B-it` is also supported on hardware:

- [BH QuietBox 2](gemma-4-12B-it_p300x2.md)

## Quickstart - Deploy gemma-4-12B-it Inference Server on BH Galaxy

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
  --volume volume_id_gemma-4-12B-it:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.25.0-978f93b-a723030 \
  --model google/gemma-4-12B-it \
  --tt-device blackhole_galaxy
```

**via run.py command**

```bash
python3 run.py --model google/gemma-4-12B-it --device blackhole_galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) |
| Model Status | 🟢 Complete |
| Max Batch Size | 128 |
| Max Context Length | 262144 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/978f93b/models/tt_transformers) |
| tt-metal Commit | `978f93b` |
| vLLM Commit | `a723030` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.25.0-978f93b-a723030` |
