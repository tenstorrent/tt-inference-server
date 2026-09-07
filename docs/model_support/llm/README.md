# LLM Models

This page lists all supported large language model models and their device compatibility.

[Search other models by model type](../../../README.md#models-by-model-type)

## Supported Models

Models with status: TOP_PERF, COMPLETE, or FUNCTIONAL.

| Model Name | [Dual WH Galaxy](https://tenstorrent.com/hardware/galaxy) | [WH Galaxy](https://tenstorrent.com/hardware/galaxy) | [BH LoudBox](https://tenstorrent.com/hardware/tt-loudbox) | [BH P300](https://tenstorrent.com/hardware/blackhole) | [BH QuietBox 2](https://tenstorrent.com/hardware/tt-quietbox) | [P150](https://tenstorrent.com/hardware/blackhole) | [WH LoudBox/QuietBox](https://tenstorrent.com/hardware/tt-loudbox) | [N150](https://tenstorrent.com/hardware/wormhole) | [N300](https://tenstorrent.com/hardware/wormhole) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Llama-3.3-70B-Instruct](Llama-3.3-70B-Instruct_galaxy.md) | - | [🟢 Complete](Llama-3.3-70B-Instruct_galaxy.md) | [🟡 Functional](Llama-3.3-70B-Instruct_p150x8.md) | - | [🟡 Functional](Llama-3.3-70B-Instruct_p300x2.md) | - | - | - | - |
| [Qwen3-32B](Qwen3-32B_galaxy.md) | - | [🟢 Complete](Qwen3-32B_galaxy.md) | [🟡 Functional](Qwen3-32B_p150x8.md) | - | [🟡 Functional](Qwen3-32B_p300x2.md) | - | - | - | - |
| [Llama-3.1-8B](Llama-3.1-8B_galaxy.md) | - | [🟡 Functional](Llama-3.1-8B_galaxy.md) | [🟡 Functional](Llama-3.1-8B_p150x8.md) | [🟡 Functional](Llama-3.1-8B_p300.md) | [🟡 Functional](Llama-3.1-8B_p300x2.md) | [🛠️ Experimental](Llama-3.1-8B_p150.md) | - | - | - |
| [Llama-3.2-3B](Llama-3.2-3B_t3k.md) | - | - | - | - | - | - | [🟡 Functional](Llama-3.2-3B_t3k.md) | [🟡 Functional](Llama-3.2-3B_n150.md) | [🟡 Functional](Llama-3.2-3B_n300.md) |
| [Qwen2.5-72B](Qwen2.5-72B_galaxy.md) | - | [🟡 Functional](Qwen2.5-72B_galaxy.md) | - | - | - | - | - | - | - |
| [Qwen3-8B](Qwen3-8B_galaxy.md) | - | [🟡 Functional](Qwen3-8B_galaxy.md) | - | - | - | - | - | - | - |
| [QwQ-32B](QwQ-32B_galaxy.md) | - | [🟡 Functional](QwQ-32B_galaxy.md) | - | - | - | - | - | - | - |

## Experimental Models

Models with EXPERIMENTAL status are under active development and may have stability or performance issues.

| Model Name | [Dual WH Galaxy](https://tenstorrent.com/hardware/galaxy) | [WH Galaxy](https://tenstorrent.com/hardware/galaxy) | [BH LoudBox](https://tenstorrent.com/hardware/tt-loudbox) | [BH P300](https://tenstorrent.com/hardware/blackhole) | [BH QuietBox 2](https://tenstorrent.com/hardware/tt-quietbox) | [P150](https://tenstorrent.com/hardware/blackhole) | [WH LoudBox/QuietBox](https://tenstorrent.com/hardware/tt-loudbox) | [N150](https://tenstorrent.com/hardware/wormhole) | [N300](https://tenstorrent.com/hardware/wormhole) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gpt-oss-120b](gpt-oss-120b_galaxy.md) | - | [🛠️ Experimental](gpt-oss-120b_galaxy.md) | - | - | [🛠️ Experimental](gpt-oss-120b_p300x2.md) | - | - | - | - |
| [AFM-4.5B](AFM-4.5B_t3k.md) | - | - | - | - | - | - | [🛠️ Experimental](AFM-4.5B_t3k.md) | - | [🛠️ Experimental](AFM-4.5B_n300.md) |
| [DeepSeek-R1-0528](DeepSeek-R1-0528_dual_galaxy.md) | [🛠️ Experimental](DeepSeek-R1-0528_dual_galaxy.md) | - | - | - | - | - | - | - | - |
| [diffusiongemma-26B-A4B-it](diffusiongemma-26B-A4B-it_p300x2.md) | - | - | - | - | [🛠️ Experimental](diffusiongemma-26B-A4B-it_p300x2.md) | - | - | - | - |
| [Falcon3-7B-Instruct](Falcon3-7B-Instruct_p150.md) | - | - | - | - | - | [🛠️ Experimental](Falcon3-7B-Instruct_p150.md) | - | - | - |
| [gemma-4-31B-it](gemma-4-31B-it_p300x2.md) | - | - | - | - | [🛠️ Experimental](gemma-4-31B-it_p300x2.md) | - | - | - | - |
| [Qwen3.6-27B](Qwen3.6-27B_p300x2.md) | - | - | - | - | [🛠️ Experimental](Qwen3.6-27B_p300x2.md) | - | - | - | - |
