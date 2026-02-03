# wan2.2-t2v-a14b-diffusers Tenstorrent Support on Tenstorrent Galaxy

wan2.2-t2v-a14b-diffusers is also supported on:

- [TT-LoudBox](wan2.2-t2v-a14b-diffusers_t3k.md)

#### Back links

- [Tenstorrent Galaxy details](https://tenstorrent.com/hardware/galaxy)
- [Search other video models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

## Quickstart - Deploy wan2.2-t2v-a14b-diffusers Inference Server on Tenstorrent Galaxy

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model wan2.2-t2v-a14b-diffusers --device galaxy --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c180ef7/models/tt_transformers) |
| tt-metal Commit | `c180ef7` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-c180ef7` |
