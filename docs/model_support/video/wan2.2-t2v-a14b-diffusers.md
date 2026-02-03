# wan2.2-t2v-a14b-diffusers Tenstorrent Support

## Run wan2.2-t2v-a14b-diffusers on TT-LoudBox

[Video Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model wan2.2-t2v-a14b-diffusers --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c180ef7/models/tt_transformers) |
| tt-metal Commit | `c180ef7` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-c180ef7` |

## Run wan2.2-t2v-a14b-diffusers on Tenstorrent Galaxy

[Video Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model wan2.2-t2v-a14b-diffusers --device galaxy --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) |
| Model Status | 🟢 Complete |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/c180ef7/models/tt_transformers) |
| tt-metal Commit | `c180ef7` |
| Docker Image | `ghcr.io/tenstorrent/tt-media-inference-server:0.8.0-c180ef7` |
