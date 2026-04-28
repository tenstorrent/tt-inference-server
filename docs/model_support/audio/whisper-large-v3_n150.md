# whisper-large-v3 Tenstorrent Support on N150/N300

Supported weights variants for this model implementation are:

- `whisper-large-v3`: [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) **(default)** 
- `distil-large-v3`: [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)

To use non-default weights, replace `whisper-large-v3` in commands below.

#### Useful links

- [N150/N300 details](https://tenstorrent.com/hardware/wormhole)
- [Search other audio models](./README.md)
- [Search other models by model type](../../../README.md#models-by-model-type)

`whisper-large-v3` is also supported on hardware:

- [WH Galaxy](whisper-large-v3_galaxy.md)
- [P100/P150](whisper-large-v3_p100.md)
- [WH LoudBox/QuietBox](whisper-large-v3_t3k.md)

## Quickstart - Deploy whisper-large-v3 Inference Server on n150

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

This model is supported by [tt-media-server](../../../tt-media-server/README.md) inference engine.

**via run.py command**

```bash
python3 run.py --model whisper-large-v3 --device n150 --workflow server --docker-server
```
For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide.

## Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [whisper](https://github.com/tenstorrent/tt-metal/tree/a8c5af0/models/demos/whisper) |
| tt-metal Commit | `a8c5af0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.11.1-a8c5af0` |

---

## N300 Configuration

### Quickstart - Deploy on n300

**via run.py command**

```bash
python3 run.py --model whisper-large-v3 --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [whisper](https://github.com/tenstorrent/tt-metal/tree/a8c5af0/models/demos/whisper) |
| tt-metal Commit | `a8c5af0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.11.1-a8c5af0` |
