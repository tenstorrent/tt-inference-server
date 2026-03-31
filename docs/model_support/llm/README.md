# LLM Models

This page lists all supported large language model models and their device compatibility.

[Search other models by model type](../../../README.md#models-by-model-type)

## Supported Models

Models with status: TOP_PERF, COMPLETE, or FUNCTIONAL.

| Model Name | [Dual WH Galaxy](https://tenstorrent.com/hardware/galaxy) | [Quad WH Galaxy](https://tenstorrent.com/hardware/galaxy) | [WH Galaxy](https://tenstorrent.com/hardware/galaxy) | [BH LoudBox](https://tenstorrent.com/hardware/tt-loudbox) | [BH 4xP150](https://tenstorrent.com/hardware/tt-quietbox) | [P100/P150](https://tenstorrent.com/hardware/blackhole) | [WH LoudBox/QuietBox](https://tenstorrent.com/hardware/tt-loudbox) | [N150/N300](https://tenstorrent.com/hardware/wormhole) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Llama-3.3-70B-Instruct](Llama-3.3-70B-Instruct_galaxy.md) | - | - | [🟢 Complete](Llama-3.3-70B-Instruct_galaxy.md) | [🟡 Functional](Llama-3.3-70B-Instruct_p150x8.md) | [🟡 Functional](Llama-3.3-70B-Instruct_p150x4.md) | - | [🟡 Functional](Llama-3.3-70B-Instruct_t3k.md) | - |
| [Qwen3-32B](Qwen3-32B_galaxy.md) | - | - | [🟢 Complete](Qwen3-32B_galaxy.md) | [🟡 Functional](Qwen3-32B_p150x8.md) | - | - | [🟡 Functional](Qwen3-32B_t3k.md) | - |
| [Llama-3.1-8B](Llama-3.1-8B_galaxy.md) | - | - | [🟡 Functional](Llama-3.1-8B_galaxy.md) | [🟡 Functional](Llama-3.1-8B_p150x8.md) | [🟢 Complete](Llama-3.1-8B_p150x4.md) | [🛠️ Experimental](Llama-3.1-8B_p100.md) | [🟢 Complete](Llama-3.1-8B_t3k.md) | [🟢 Complete](Llama-3.1-8B_n150.md) |
| [Mistral-7B-Instruct-v0.3](Mistral-7B-Instruct-v0.3_t3k.md) | - | - | - | - | - | - | [🟢 Complete](Mistral-7B-Instruct-v0.3_t3k.md) | [🟢 Complete](Mistral-7B-Instruct-v0.3_n150.md) |
| [Llama-3.2-1B](Llama-3.2-1B_t3k.md) | - | - | - | - | - | - | [🟡 Functional](Llama-3.2-1B_t3k.md) | [🟡 Functional](Llama-3.2-1B_n150.md) |
| [Llama-3.2-3B](Llama-3.2-3B_t3k.md) | - | - | - | - | - | - | [🟡 Functional](Llama-3.2-3B_t3k.md) | [🟡 Functional](Llama-3.2-3B_n150.md) |
| [Qwen2.5-72B](Qwen2.5-72B_galaxy.md) | - | - | [🟡 Functional](Qwen2.5-72B_galaxy.md) | - | - | - | [🟡 Functional](Qwen2.5-72B_t3k.md) | - |
| [Qwen3-8B](Qwen3-8B_galaxy.md) | - | - | [🟡 Functional](Qwen3-8B_galaxy.md) | - | - | - | [🟡 Functional](Qwen3-8B_t3k.md) | [🟡 Functional](Qwen3-8B_n150.md) |
| [QwQ-32B](QwQ-32B_galaxy.md) | - | - | [🟡 Functional](QwQ-32B_galaxy.md) | - | - | - | [🟡 Functional](QwQ-32B_t3k.md) | - |

## Experimental Models

Models with EXPERIMENTAL status are under active development and may have stability or performance issues.

| Model Name | [Dual WH Galaxy](https://tenstorrent.com/hardware/galaxy) | [Quad WH Galaxy](https://tenstorrent.com/hardware/galaxy) | [WH Galaxy](https://tenstorrent.com/hardware/galaxy) | [BH LoudBox](https://tenstorrent.com/hardware/tt-loudbox) | [BH 4xP150](https://tenstorrent.com/hardware/tt-quietbox) | [P100/P150](https://tenstorrent.com/hardware/blackhole) | [WH LoudBox/QuietBox](https://tenstorrent.com/hardware/tt-loudbox) | [N150/N300](https://tenstorrent.com/hardware/wormhole) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gpt-oss-120b](gpt-oss-120b_galaxy.md) | - | - | [🛠️ Experimental](gpt-oss-120b_galaxy.md) | - | - | - | [🛠️ Experimental](gpt-oss-120b_t3k.md) | - |
| [gpt-oss-20b](gpt-oss-20b_galaxy.md) | - | - | [🛠️ Experimental](gpt-oss-20b_galaxy.md) | - | - | - | [🛠️ Experimental](gpt-oss-20b_t3k.md) | - |
| [AFM-4.5B](AFM-4.5B_t3k.md) | - | - | - | - | - | - | [🛠️ Experimental](AFM-4.5B_t3k.md) | [🛠️ Experimental](AFM-4.5B_n150.md) |
| [DeepSeek-R1-0528](DeepSeek-R1-0528_dual_galaxy.md) | [🛠️ Experimental](DeepSeek-R1-0528_dual_galaxy.md) | [🛠️ Experimental](DeepSeek-R1-0528_quad_galaxy.md) | [🛠️ Experimental](DeepSeek-R1-0528_galaxy.md) | - | - | - | - | - |
| [gemma-3-1b-it](gemma-3-1b-it_n150.md) | - | - | - | - | - | - | - | [🛠️ Experimental](gemma-3-1b-it_n150.md) |
| [Qwen2.5-7B](Qwen2.5-7B_n150.md) | - | - | - | - | - | - | - | [🛠️ Experimental](Qwen2.5-7B_n150.md) |
| [Qwen2.5-Coder-32B-Instruct](Qwen2.5-Coder-32B-Instruct_galaxy.md) | - | - | [🛠️ Experimental](Qwen2.5-Coder-32B-Instruct_galaxy.md) | - | - | - | [🛠️ Experimental](Qwen2.5-Coder-32B-Instruct_t3k.md) | - |
| [Qwen3-4B](Qwen3-4B_n150.md) | - | - | - | - | - | - | - | [🛠️ Experimental](Qwen3-4B_n150.md) |
