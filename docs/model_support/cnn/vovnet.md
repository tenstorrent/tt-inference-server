# vovnet Tenstorrent Support

## Run vovnet on n150

[CNN Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model vovnet --device n150 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [vovnet](https://huggingface.co/vovnet) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/2496be4/models/tt_transformers) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:a9b09e0b611da6deb4d8972e8296148fd864e5fd_98dcf62_60920940673` |

## Run vovnet on n300

[CNN Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

**via run.py command**

```bash
python3 run.py --model vovnet --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [vovnet](https://huggingface.co/vovnet) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 1 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/2496be4/models/tt_transformers) |
| tt-metal Commit | `2496be4` |
| Docker Image | `ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-forge:a9b09e0b611da6deb4d8972e8296148fd864e5fd_98dcf62_60920940673` |
