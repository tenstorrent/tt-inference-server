[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tenstorrent/tt-inference-server)

# tt-inference-server

`tt-inference-server` is the fastest way to deploy and test models for serving inference on Tenstorrent hardware.

## Quickstart guide

### Prequisites

This guide assumes you have already:
1. Set up Tenstorrent hardware
2. Installed Tenstorrent System Software Stack:
    - Firmware
    - KMD
    - Hugepages

For these steps please see https://docs.tenstorrent.com/index.html

### Tenstorrent E2E Inference Model Statuses

Model implementations supported in tt-inference-server have one of the following statuses:
- 🟢 Complete: passing accuracy evals, performance is optimized on Tenstorrent hardware.
- 🟡 Functional: passing accuracy evals, performance is order-of-magnitude ok on Tenstorrent hardware.
- 🛠️ Experimental: Work in progess, added for developer testing. List of experimental models at [docs/experimental_models.md](docs/experimental_models.md)

Click on the `Model Name` name in the Model Support Tables below to be linked to the quickstart model inference deployment guide. 

The table is broken out by Tenstorrent E2E Inference Model Status, then sorted alphanumerically by Model Name.

**Note:** make sure the `Hardware` listed matches your Tenstorrent hardware you are deploying on.

## Model Support Tables

<!-- COMPLETE_MODELS_START -->
### 🟢 Complete Models

| Model Name | Hardware | Model Weights | tt-metal commit | Docker Image |
|-------|---------------|----------|-----------------|---------------|
| [Llama-3.1-8B](vllm-tt-metal-llama3/README.md) | [BH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (P150X8) | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)<br/>[meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [55fd115](https://github.com/tenstorrent/tt-metal/tree/55fd115/models/tt_transformers) | [0.8.0-55fd115-aa4ae1e](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-55fd115-aa4ae1e) |
| [Llama-3.1-8B](vllm-tt-metal-llama3/README.md) | [Galaxy](https://tenstorrent.com/hardware/galaxy) | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)<br/>[meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [a9b09e0](https://github.com/tenstorrent/tt-metal/tree/a9b09e0/models/tt_transformers) | [0.8.0-a9b09e0-a186bf4](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-a9b09e0-a186bf4) |
| [Llama-3.2-11B-Vision](vllm-tt-metal-llama3/README.md) | [n300](https://tenstorrent.com/hardware/wormhole), [WH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)/[WH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (T3K) | [meta-llama/Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)<br/>[meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) | [v0.61.1-rc1](https://github.com/tenstorrent/tt-metal/tree/v0.61.1-rc1/models/tt_transformers) | [0.8.0-v0.61.1-rc1-5cb](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-v0.61.1-rc1-5cbc982) |
| [Llama-3.2-1B](vllm-tt-metal-llama3/README.md) | [n300](https://tenstorrent.com/hardware/wormhole), [WH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)/[WH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (T3K), [n150](https://tenstorrent.com/hardware/wormhole) | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)<br/>[meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | [9b67e09](https://github.com/tenstorrent/tt-metal/tree/9b67e09/models/tt_transformers) | [0.8.0-9b67e09-a91b644](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-9b67e09-a91b644) |
| [Llama-3.2-3B](vllm-tt-metal-llama3/README.md) | [n300](https://tenstorrent.com/hardware/wormhole), [WH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)/[WH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (T3K), [n150](https://tenstorrent.com/hardware/wormhole) | [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)<br/>[meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [20edc39](https://github.com/tenstorrent/tt-metal/tree/20edc39/models/tt_transformers) | [0.8.0-20edc39-03cb300](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-20edc39-03cb300) |
| [Llama-3.2-90B-Vision](vllm-tt-metal-llama3/README.md) | [WH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)/[WH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (T3K) | [meta-llama/Llama-3.2-90B-Vision](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision)<br/>[meta-llama/Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct) | [v0.61.1-rc1](https://github.com/tenstorrent/tt-metal/tree/v0.61.1-rc1/models/tt_transformers) | [0.8.0-v0.61.1-rc1-5cb](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-v0.61.1-rc1-5cbc982) |
| [Llama-3.3-70B-Instruct](vllm-tt-metal-llama3/README.md) | [WH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)/[WH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (T3K) | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)<br/>[meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)<br/>[meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)<br/>[deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | [9b67e09](https://github.com/tenstorrent/tt-metal/tree/9b67e09/models/tt_transformers) | [0.8.0-9b67e09-a91b644](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-9b67e09-a91b644) |
| [Llama-3.3-70B-Instruct](vllm-tt-metal-llama3/README.md) | [BH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox) (P150X4) | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)<br/>[meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)<br/>[meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)<br/>[deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | [55fd115](https://github.com/tenstorrent/tt-metal/tree/55fd115/models/tt_transformers) | [0.8.0-55fd115-aa4ae1e](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-55fd115-aa4ae1e) |
| [Llama-3.3-70B-Instruct](vllm-tt-metal-llama3/README.md) | [BH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (P150X8) | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)<br/>[meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)<br/>[meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)<br/>[deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | [55fd115](https://github.com/tenstorrent/tt-metal/tree/55fd115/models/tt_transformers) | [0.8.0-55fd115-aa4ae1e](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-55fd115-aa4ae1e) |
| [Qwen2.5-72B](vllm-tt-metal-llama3/README.md) | [WH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)/[WH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (T3K), [Galaxy](https://tenstorrent.com/hardware/galaxy) | [Qwen/Qwen2.5-72B](https://huggingface.co/Qwen/Qwen2.5-72B)<br/>[Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | [13f44c5](https://github.com/tenstorrent/tt-metal/tree/13f44c5/models/tt_transformers) | [0.8.0-13f44c5-0edd242](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-13f44c5-0edd242) |
| [Qwen2.5-VL-72B-Instruct](vllm-tt-metal-llama3/README.md) | [WH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)/[WH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (T3K) | [Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) | [c18569e](https://github.com/tenstorrent/tt-metal/tree/c18569e/models/tt_transformers) | [0.8.0-c18569e-b2894d3](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-c18569e-b2894d3) |
| [Qwen3-32B](vllm-tt-metal-llama3/README.md) | [WH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)/[WH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (T3K) | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | [e95ffa5](https://github.com/tenstorrent/tt-metal/tree/e95ffa5/models/tt_transformers) | [0.8.0-e95ffa5-48eba14](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e95ffa5-48eba14) |
| [Qwen3-32B](vllm-tt-metal-llama3/README.md) | [BH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (P150X8) | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | [55fd115](https://github.com/tenstorrent/tt-metal/tree/55fd115/models/tt_transformers) | [0.8.0-55fd115-aa4ae1e](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-55fd115-aa4ae1e) |
| [Qwen3-8B](vllm-tt-metal-llama3/README.md) | [n150](https://tenstorrent.com/hardware/wormhole), [n300](https://tenstorrent.com/hardware/wormhole), [WH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)/[WH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (T3K), [Galaxy](https://tenstorrent.com/hardware/galaxy) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | [e95ffa5](https://github.com/tenstorrent/tt-metal/tree/e95ffa5/models/tt_transformers) | [0.8.0-e95ffa5-48eba14](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e95ffa5-48eba14) |
| [QwQ-32B](vllm-tt-metal-llama3/README.md) | [WH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)/[WH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (T3K), [Galaxy](https://tenstorrent.com/hardware/galaxy) | [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) | [e95ffa5](https://github.com/tenstorrent/tt-metal/tree/e95ffa5/models/tt_transformers) | [0.8.0-e95ffa5-48eba14](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e95ffa5-48eba14) |
<!-- FUNCTIONAL_MODELS_END -->

# Workflow automation in tt-inference-server

For details on the workflow automation for:
- deploying inference servers
- running E2E performance benchmarks
- running accuracy evals

See:
- [Model Readiness Workflows User Guide](docs/workflows_user_guide.md).
- [workflows/README.md](workflows/README.md)

## Benchmarking 

For more details see [benchmarking/README.md](benchmarking/README.md)

## Evals

For more details see [evals/README.md](evals/README.md)

## Development

Developer documentation: [docs/README.md](docs/README.md)

Release documentation: [scripts/release/README.md](scripts/release/README.md)

If you encounter setup or stability problems with any model please [file an issue](https://github.com/tenstorrent/tt-inference-server/issues/new?template=Blank+issue) and our team will address it.
