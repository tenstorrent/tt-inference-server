# whisper-large-v3 Tenstorrent Support

## Run whisper-large-v3 on n150

[Audio Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `whisper-large-v3`, the following weights are supported as well:

- `distil-large-v3`

To use these weights simply swap `whisper-large-v3` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model whisper-large-v3 --device n150 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [whisper](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/demos/whisper) |
| tt-metal Commit | `a9b09e0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-a9b09e0` |

## Run whisper-large-v3 on TT-LoudBox

[Audio Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `whisper-large-v3`, the following weights are supported as well:

- `distil-large-v3`

To use these weights simply swap `whisper-large-v3` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model whisper-large-v3 --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) |
| Model Status | 🟢 Complete |
| Max Batch Size | 4 |
| Implementation Code | [whisper](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/demos/whisper) |
| tt-metal Commit | `a9b09e0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-a9b09e0` |

## Run whisper-large-v3 on Tenstorrent Galaxy

[Audio Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `whisper-large-v3`, the following weights are supported as well:

- `distil-large-v3`

To use these weights simply swap `whisper-large-v3` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model whisper-large-v3 --device galaxy --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3), [distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) |
| Model Status | 🟢 Complete |
| Max Batch Size | 32 |
| Implementation Code | [whisper](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/demos/whisper) |
| tt-metal Commit | `a9b09e0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-a9b09e0` |
