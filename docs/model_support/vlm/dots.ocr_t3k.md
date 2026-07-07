# dots.ocr Tenstorrent Support on WH LoudBox/QuietBox

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other vlm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy dots.ocr Inference Server on WH LoudBox/QuietBox

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
  --volume volume_id_dots.ocr:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-tt-symbiote-src-release-ubuntu-22.04-amd64:0.17.0-c09f09c35a1a-cdb395febdc9 \
  --model dots.ocr \
  --tt-device t3k
```

**via run.py command**

```bash
python3 run.py --model dots.ocr --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 8 |
| Max Context Length | 11264 |
| Implementation Code | [tt-symbiote](https://github.com/tenstorrent/tt_symbiote/tree/c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195/src/tt_symbiote) |
| tt-metal Commit | `c09f09c35a1a59a428f0e1b5cdaa8fe59fb1b195` |
| vLLM Commit | `cdb395febdc9caaeba6d82e283329366ac9c2d89` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-tt-symbiote-src-release-ubuntu-22.04-amd64:0.17.0-c09f09c35a1a-cdb395febdc9` |
