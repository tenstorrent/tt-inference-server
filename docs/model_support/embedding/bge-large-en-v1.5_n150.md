# bge-large-en-v1.5 Tenstorrent Support on N150/N300

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other embedding models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`bge-large-en-v1.5` is also supported on hardware:

- [WH Galaxy](bge-large-en-v1.5_galaxy.md)
- [WH LoudBox/QuietBox](bge-large-en-v1.5_t3k.md)

## Quickstart - Deploy bge-large-en-v1.5 Inference Server on n150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model bge-large-en-v1.5 --device n150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [tt-vllm-plugin](https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-vllm-plugin/tree/a8c5af0/tt_vllm_plugin) |
| tt-metal Commit | `a8c5af0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.2.0-2496be4518bca0a7a5b497a4cda3cfe7e2f59756` |

---

## N300 Configuration

### Quickstart - Deploy on n300

**via run.py command**

```bash
python3 run.py --model bge-large-en-v1.5 --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [tt-vllm-plugin](https://github.com/tenstorrent/tt-inference-server/tree/dev/tt-vllm-plugin/tree/a8c5af0/tt_vllm_plugin) |
| tt-metal Commit | `a8c5af0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.2.0-2496be4518bca0a7a5b497a4cda3cfe7e2f59756` |
