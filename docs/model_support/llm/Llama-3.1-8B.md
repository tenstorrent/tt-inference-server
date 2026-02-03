# Llama-3.1-8B Tenstorrent Support

## Run Llama-3.1-8B on n150

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.1-8B`, the following weights are supported as well:

- `Llama-3.1-8B-Instruct`

To use these weights simply swap `Llama-3.1-8B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device n150 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🟢 Complete |
| Max Batch Size | 32 |
| Max Context Length | 65536 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/25305db/models/tt_transformers) |
| tt-metal Commit | `25305db` |
| vLLM Commit | `6e67d2d` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-25305db-6e67d2d` |

## Run Llama-3.1-8B on n300

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.1-8B`, the following weights are supported as well:

- `Llama-3.1-8B-Instruct`

To use these weights simply swap `Llama-3.1-8B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device n300 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🟢 Complete |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/25305db/models/tt_transformers) |
| tt-metal Commit | `25305db` |
| vLLM Commit | `6e67d2d` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-25305db-6e67d2d` |

## Run Llama-3.1-8B on TT-LoudBox

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.1-8B`, the following weights are supported as well:

- `Llama-3.1-8B-Instruct`

To use these weights simply swap `Llama-3.1-8B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🟢 Complete |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/25305db/models/tt_transformers) |
| tt-metal Commit | `25305db` |
| vLLM Commit | `6e67d2d` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-25305db-6e67d2d` |

## Run Llama-3.1-8B on Tenstorrent Galaxy

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.1-8B`, the following weights are supported as well:

- `Llama-3.1-8B-Instruct`

To use these weights simply swap `Llama-3.1-8B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device galaxy --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 128 |
| Max Context Length | 65536 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/tt_transformers) |
| tt-metal Commit | `a9b09e0` |
| vLLM Commit | `a186bf4` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-a9b09e0-a186bf4` |

## Run Llama-3.1-8B on Tenstorrent Galaxy (GALAXY_T3K)

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.1-8B`, the following weights are supported as well:

- `Llama-3.1-8B-Instruct`

To use these weights simply swap `Llama-3.1-8B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device galaxy_t3k --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/tt_transformers) |
| tt-metal Commit | `a9b09e0` |
| vLLM Commit | `a186bf4` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-a9b09e0-a186bf4` |

## Run Llama-3.1-8B on p100

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.1-8B`, the following weights are supported as well:

- `Llama-3.1-8B-Instruct`

To use these weights simply swap `Llama-3.1-8B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device p100 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 65536 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/55fd115/models/tt_transformers) |
| tt-metal Commit | `55fd115` |
| vLLM Commit | `aa4ae1e` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-55fd115-aa4ae1e` |

## Run Llama-3.1-8B on p150

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.1-8B`, the following weights are supported as well:

- `Llama-3.1-8B-Instruct`

To use these weights simply swap `Llama-3.1-8B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device p150 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🛠️ Experimental |
| Max Batch Size | 32 |
| Max Context Length | 65536 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/55fd115/models/tt_transformers) |
| tt-metal Commit | `55fd115` |
| vLLM Commit | `aa4ae1e` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-55fd115-aa4ae1e` |

## Run Llama-3.1-8B on 4xp150

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.1-8B`, the following weights are supported as well:

- `Llama-3.1-8B-Instruct`

To use these weights simply swap `Llama-3.1-8B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device p150x4 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🟢 Complete |
| Max Batch Size | 128 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/55fd115/models/tt_transformers) |
| tt-metal Commit | `55fd115` |
| vLLM Commit | `aa4ae1e` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-55fd115-aa4ae1e` |

## Run Llama-3.1-8B on 8xp150

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.1-8B`, the following weights are supported as well:

- `Llama-3.1-8B-Instruct`

To use these weights simply swap `Llama-3.1-8B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device p150x8 --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🟡 Functional |
| Max Batch Size | 256 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/55fd115/models/tt_transformers) |
| tt-metal Commit | `55fd115` |
| vLLM Commit | `aa4ae1e` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-55fd115-aa4ae1e` |

## Run Llama-3.1-8B on GPU

[LLM Model Support Table](README.md)

### Quickstart - Deploy Inference Server

See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues.

The default model weights for this implementation is `Llama-3.1-8B`, the following weights are supported as well:

- `Llama-3.1-8B-Instruct`

To use these weights simply swap `Llama-3.1-8B` for your desired weights in commands below.

**via run.py command**

```bash
python3 run.py --model Llama-3.1-8B --device gpu --workflow server --docker-server
```

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Weights | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Model Status | 🟢 Complete |
| Max Batch Size | 32 |
| Max Context Length | 131072 |
| Implementation Code | [tt-transformers](https://github.com/tenstorrent/tt-metal/tree/25305db/models/tt_transformers) |
| tt-metal Commit | `25305db` |
| vLLM Commit | `6e67d2d` |
| Docker Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-25305db-6e67d2d` |
