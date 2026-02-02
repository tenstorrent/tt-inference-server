# LLM Models

This page lists all supported large language models and their device compatibility.

[Search model by model type](../../README.md#models-by-model-type)

## Supported Models

Models with status: TOP_PERF, COMPLETE, or FUNCTIONAL.

| Model Name | N150 | N300 | T3K | GALAXY | GALAXY_T3K | P100 | P150 | P150X4 | P150X8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [gpt-oss-120b](llm/gpt-oss-120b.md) | - | - | [🟡 Functional](llm/gpt-oss-120b.md#run-gpt-oss-120b-on-tt-loudbox) | [🟡 Functional](llm/gpt-oss-120b.md#run-gpt-oss-120b-on-tenstorrent-galaxy) | - | - | - | - | - |
| [gpt-oss-20b](llm/gpt-oss-20b.md) | - | - | [🟡 Functional](llm/gpt-oss-20b.md#run-gpt-oss-20b-on-tt-loudbox) | [🟡 Functional](llm/gpt-oss-20b.md#run-gpt-oss-20b-on-tenstorrent-galaxy) | [🟡 Functional](llm/gpt-oss-20b.md#run-gpt-oss-20b-on-tenstorrent-galaxy-galaxy_t3k) | - | - | - | - |
| [Llama-3.1-8B](llm/Llama-3.1-8B.md) | [🟢 Complete](llm/Llama-3.1-8B.md#run-llama-31-8b-on-n150) | [🟢 Complete](llm/Llama-3.1-8B.md#run-llama-31-8b-on-n300) | [🟢 Complete](llm/Llama-3.1-8B.md#run-llama-31-8b-on-tt-loudbox) | [🟡 Functional](llm/Llama-3.1-8B.md#run-llama-31-8b-on-tenstorrent-galaxy) | [🟡 Functional](llm/Llama-3.1-8B.md#run-llama-31-8b-on-tenstorrent-galaxy-galaxy_t3k) | [🛠️ Experimental](llm/Llama-3.1-8B.md#run-llama-31-8b-on-p100) | [🛠️ Experimental](llm/Llama-3.1-8B.md#run-llama-31-8b-on-p150) | [🟢 Complete](llm/Llama-3.1-8B.md#run-llama-31-8b-on-4xp150) | [🟡 Functional](llm/Llama-3.1-8B.md#run-llama-31-8b-on-8xp150) |
| [Llama-3.2-1B](llm/Llama-3.2-1B.md) | [🟡 Functional](llm/Llama-3.2-1B.md#run-llama-32-1b-on-n150) | [🟡 Functional](llm/Llama-3.2-1B.md#run-llama-32-1b-on-n300) | [🟡 Functional](llm/Llama-3.2-1B.md#run-llama-32-1b-on-tt-loudbox) | - | - | - | - | - | - |
| [Llama-3.2-3B](llm/Llama-3.2-3B.md) | [🟡 Functional](llm/Llama-3.2-3B.md#run-llama-32-3b-on-n150) | [🟡 Functional](llm/Llama-3.2-3B.md#run-llama-32-3b-on-n300) | [🟡 Functional](llm/Llama-3.2-3B.md#run-llama-32-3b-on-tt-loudbox) | - | - | - | - | - | - |
| [Llama-3.3-70B-Instruct](llm/Llama-3.3-70B-Instruct.md) | - | - | [🟡 Functional](llm/Llama-3.3-70B-Instruct.md#run-llama-33-70b-instruct-on-tt-loudbox) | [🟢 Complete](llm/Llama-3.3-70B-Instruct.md#run-llama-33-70b-instruct-on-tenstorrent-galaxy) | [🟡 Functional](llm/Llama-3.3-70B-Instruct.md#run-llama-33-70b-instruct-on-tenstorrent-galaxy-galaxy_t3k) | - | - | [🟡 Functional](llm/Llama-3.3-70B-Instruct.md#run-llama-33-70b-instruct-on-4xp150) | [🟡 Functional](llm/Llama-3.3-70B-Instruct.md#run-llama-33-70b-instruct-on-8xp150) |
| [Mistral-7B-Instruct-v0.3](llm/Mistral-7B-Instruct-v0.3.md) | [🟢 Complete](llm/Mistral-7B-Instruct-v0.3.md#run-mistral-7b-instruct-v03-on-n150) | [🟢 Complete](llm/Mistral-7B-Instruct-v0.3.md#run-mistral-7b-instruct-v03-on-n300) | [🟢 Complete](llm/Mistral-7B-Instruct-v0.3.md#run-mistral-7b-instruct-v03-on-tt-loudbox) | - | - | - | - | - | - |
| [Qwen2.5-72B](llm/Qwen2.5-72B.md) | - | - | [🟡 Functional](llm/Qwen2.5-72B.md#run-qwen25-72b-on-tt-loudbox) | [🟡 Functional](llm/Qwen2.5-72B.md#run-qwen25-72b-on-tenstorrent-galaxy) | [🟡 Functional](llm/Qwen2.5-72B.md#run-qwen25-72b-on-tenstorrent-galaxy-galaxy_t3k) | - | - | - | - |
| [Qwen3-32B](llm/Qwen3-32B.md) | - | - | [🟡 Functional](llm/Qwen3-32B.md#run-qwen3-32b-on-tt-loudbox) | [🟢 Complete](llm/Qwen3-32B.md#run-qwen3-32b-on-tenstorrent-galaxy) | [🟡 Functional](llm/Qwen3-32B.md#run-qwen3-32b-on-tenstorrent-galaxy-galaxy_t3k) | - | - | - | [🟡 Functional](llm/Qwen3-32B.md#run-qwen3-32b-on-8xp150) |
| [Qwen3-8B](llm/Qwen3-8B.md) | [🟡 Functional](llm/Qwen3-8B.md#run-qwen3-8b-on-n150) | [🟡 Functional](llm/Qwen3-8B.md#run-qwen3-8b-on-n300) | [🟡 Functional](llm/Qwen3-8B.md#run-qwen3-8b-on-tt-loudbox) | [🟡 Functional](llm/Qwen3-8B.md#run-qwen3-8b-on-tenstorrent-galaxy) | [🟡 Functional](llm/Qwen3-8B.md#run-qwen3-8b-on-tenstorrent-galaxy-galaxy_t3k) | - | - | - | - |
| [QwQ-32B](llm/QwQ-32B.md) | - | - | [🟡 Functional](llm/QwQ-32B.md#run-qwq-32b-on-tt-loudbox) | [🟡 Functional](llm/QwQ-32B.md#run-qwq-32b-on-tenstorrent-galaxy) | [🟡 Functional](llm/QwQ-32B.md#run-qwq-32b-on-tenstorrent-galaxy-galaxy_t3k) | - | - | - | - |

## Experimental Models

Models with EXPERIMENTAL status are under active development and may have stability or performance issues.

| Model Name | N150 | N300 | T3K | GALAXY | GALAXY_T3K | P100 | P150 | P150X4 | P150X8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [AFM-4.5B](llm/AFM-4.5B.md) | - | [🛠️ Experimental](llm/AFM-4.5B.md#run-afm-45b-on-n300) | [🛠️ Experimental](llm/AFM-4.5B.md#run-afm-45b-on-tt-loudbox) | - | - | - | - | - | - |
| [DeepSeek-R1-0528](llm/DeepSeek-R1-0528.md) | - | - | - | [🛠️ Experimental](llm/DeepSeek-R1-0528.md#run-deepseek-r1-0528-on-tenstorrent-galaxy) | - | - | - | - | - |
| [gemma-3-1b-it](llm/gemma-3-1b-it.md) | [🛠️ Experimental](llm/gemma-3-1b-it.md#run-gemma-3-1b-it-on-n150) | - | - | - | - | - | - | - | - |
| [Qwen2.5-7B](llm/Qwen2.5-7B.md) | - | [🛠️ Experimental](llm/Qwen2.5-7B.md#run-qwen25-7b-on-n300) | - | - | - | - | - | - | - |
| [Qwen2.5-Coder-32B-Instruct](llm/Qwen2.5-Coder-32B-Instruct.md) | - | - | [🛠️ Experimental](llm/Qwen2.5-Coder-32B-Instruct.md#run-qwen25-coder-32b-instruct-on-tt-loudbox) | - | [🛠️ Experimental](llm/Qwen2.5-Coder-32B-Instruct.md#run-qwen25-coder-32b-instruct-on-tenstorrent-galaxy-galaxy_t3k) | - | - | - | - |
