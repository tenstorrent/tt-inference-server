# TT vLLM Plugin

A vLLM plugin that enables running Large Language Models (LLMs) on Tenstorrent hardware using the vLLM v1 architecture.

## Overview

This plugin extracts the Tenstorrent (TT) platform implementation from the [vllm fork](https://github.com/tenstorrent/vllm/tree/dev) and packages it as a standalone, installable vLLM plugin.

Limitations:
- Supports only v1 architecture
- `override_tt_config` supported by [the fork](https://github.com/tenstorrent/vllm/tree/dev) is not supported in the plugin.

## Installation

### Prerequisites

- Release version of the `tt-metal`
- Set `VLLM_USE_V1=1` environment variable

You have to be in the `tt-metal` python environment to use the plugin.

```bash
source tt-metal/python_env/bin/activate
```

### Install the Plugin

Then, you can install the plugin and its dependencies:
```bash
cd tt-vllm-plugin
pip install -e .
```

In case there are dependency conflicts run this:
```
pip install "vllm<0.11" "torch==2.7.1+cpu" "torchvision==0.22.1+cpu" "numpy==1.26.4"
```

### Verify Installation

Verify `tt-metal` is working correctly by running:
```bash
python examples/ttnn_test.py
```

Then verify vLLM plugin is installed correctly by running:
```bash
python examples/llama_3_1_8b_instruct.py
```

You should see `tt` in the list of available platform plugins.

```
INFO 11-28 13:51:29 [__init__.py:36] Available plugins for group vllm.platform_plugins:
INFO 11-28 13:51:29 [__init__.py:38] - tt -> tt_vllm_plugin:register
INFO 11-28 13:51:29 [__init__.py:41] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 11-28 13:51:29 [__init__.py:232] Platform plugin tt is activated
```

## Usage

### Starting the Server

To start the vLLM server, use the provided `serve.sh` script:

```bash
./serve.sh
```

This will start the server with the default configuration for `meta-llama/Llama-3.1-8B-Instruct` on `localhost:8000`.

### Running Online Inference Performance Benchmark

After the server is started, you can run the online inference performance test using `online_benchmark.sh`:

```bash
./online_benchmark.sh
```

This benchmark script will:
- Connect to the running server at `localhost:8000`
- Use the ShareGPT dataset for testing
- Run 500 prompts at a request rate of 2 requests per second
- Save the benchmark results to a JSON file

**Note:** Make sure the server is running before executing the benchmark script.

## How to measure performance

### Embedding models

Serve the model with the following command:
```bash
VLLM_USE_V1=1 HF_MODEL='BAAI/bge-large-en-v1.5' \
vllm serve 'BAAI/bge-large-en-v1.5' \
    --max-model-len 384 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 3072
```

Set `max-num-batched-tokens` to `max-model-len * max-num-seqs`.

Embedding benchmarks may not be supported by the vLLM version this plugin is using. This might require you to use separate `venv` for benchmarks.

Run the benchmark with the following command:
```bash
vllm bench serve \
  --model 'BAAI/bge-large-en-v1.5' \
  --backend openai-embeddings \
  --endpoint /v1/embeddings \
  --dataset-name random \
  --random-input-len 384 \
  --save-result \
  --result-dir benchmark
```

## Tested models

- meta-llama/Llama-3.1-8B-Instruct on N150
- BAAI/bge-large-en-v1.5 on N150 and Galaxy

## References

- [vLLM Plugin System Documentation](https://docs.vllm.ai/en/stable/design/plugin_system.html)
- [vLLM v1 Architecture Blog](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)