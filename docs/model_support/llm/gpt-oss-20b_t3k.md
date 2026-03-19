# gpt-oss-20b Tenstorrent Support on WH LoudBox/QuietBox

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other llm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`gpt-oss-20b` is also supported on hardware:

- [WH Galaxy](gpt-oss-20b_galaxy.md)

## Quickstart - Deploy gpt-oss-20b Inference Server on WH LoudBox/QuietBox

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
  --volume volume_id_gpt-oss-20b:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-e867533-8f36910 \
  --model gpt-oss-20b \
  --tt-device t3k
```

**via run.py command**

```bash
python3 run.py --model gpt-oss-20b --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Max Context Length | 16384 |
| Implementation Code | [gpt-oss](https://github.com/tenstorrent/tt-metal/tree/e867533/models/demos/gpt_oss) |
| tt-metal Commit | `e867533` |
| vLLM Commit | `8f36910` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-e867533-8f36910` |
