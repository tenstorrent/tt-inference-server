# speecht5_tts Tenstorrent Support

## Run speecht5_tts on n150

[TTS Model Support Table](../tts_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model speecht5_tts --device n150 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [speecht5-tts](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/experimental/speecht5_tts) |
| tt-metal Commit | `a9b09e0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-a9b09e0` |

## Run speecht5_tts on n300

[TTS Model Support Table](../tts_models.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model speecht5_tts --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [speecht5-tts](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/experimental/speecht5_tts) |
| tt-metal Commit | `a9b09e0` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-a9b09e0` |
