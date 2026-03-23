# Molmo-7B-O-0924 Tenstorrent Support on WH LoudBox/QuietBox

Supported weights variants for this model implementation are:

- `Molmo-7B-O-0924`: [allenai/Molmo-7B-O-0924](https://huggingface.co/allenai/Molmo-7B-O-0924) **(default)**

#### Useful links

- [WH LoudBox/QuietBox details](https://tenstorrent.com/hardware/tt-loudbox)
- [Search other vlm models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy Molmo-7B-O-0924 Inference Server on WH LoudBox/QuietBox

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
  --volume volume_id_Molmo-7B-O-0924:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-1a4cd53-b2894d3 \
  --model Molmo-7B-O-0924 \
  --tt-device t3k
```

**via run.py command**

```bash
python3 run.py --model Molmo-7B-O-0924 --device t3k --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [allenai/Molmo-7B-O-0924](https://huggingface.co/allenai/Molmo-7B-O-0924) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Max Context Length | 4096 |
| Supported Modalities | text, image, video |
| Implementation Code | [tt-metal](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/molmo2) |
| tt-metal Commit | `1a4cd53` |
| vLLM Commit | `b2894d3` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-1a4cd53-b2894d3` |

## Notes

- Molmo2 (Molmo-7B-O-0924) is a Vision-Language Model from Allen Institute for AI
- Runs on T3K (8 Wormhole devices, 1x8 mesh)
- Supports text-only, image, and video inputs
- Video processing: 8 frames extracted per video
- Requires `trust_remote_code=True` for HuggingFace model loading
