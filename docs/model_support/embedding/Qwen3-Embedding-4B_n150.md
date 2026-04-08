# Qwen3-Embedding-4B Tenstorrent Support on N150/N300

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other embedding models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`Qwen3-Embedding-4B` is also supported on hardware:

- [WH Galaxy](Qwen3-Embedding-4B_galaxy.md)
- [WH LoudBox/QuietBox](Qwen3-Embedding-4B_t3k.md)

## Quickstart - Deploy Qwen3-Embedding-4B Inference Server on n150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server (forge plugin)](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model Qwen3-Embedding-4B --device n150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [forge-vllm-plugin](https://github.com/tenstorrent/tt-xla/tree/main/tree/2496be4/integrations/vllm_plugin) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:a9b09e0b611da6deb4d8972e8296148fd864e5fd_98dcf62_60920940673` |

---

## N300 Configuration

### Quickstart - Deploy on n300

**via run.py command**

```bash
python3 run.py --model Qwen3-Embedding-4B --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [forge-vllm-plugin](https://github.com/tenstorrent/tt-xla/tree/main/tree/2496be4/integrations/vllm_plugin) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:a9b09e0b611da6deb4d8972e8296148fd864e5fd_98dcf62_60920940673` |
