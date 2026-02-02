# VLM Models

This page lists all supported vision-language models and their device compatibility.

[Search model by model type](../../README.md#models-by-model-type)

## Supported Models

Models with status: TOP_PERF, COMPLETE, or FUNCTIONAL.

| Model Name | N150 | N300 | T3K | GALAXY | GALAXY_T3K |
| --- | --- | --- | --- | --- | --- |
| [Llama-3.2-11B-Vision](vlm/Llama-3.2-11B-Vision.md) | - | [🟡 Functional](vlm/Llama-3.2-11B-Vision.md#run-llama-32-11b-vision-on-n300) | [🟡 Functional](vlm/Llama-3.2-11B-Vision.md#run-llama-32-11b-vision-on-tt-loudbox) | - | - |
| [Llama-3.2-90B-Vision](vlm/Llama-3.2-90B-Vision.md) | - | - | [🟡 Functional](vlm/Llama-3.2-90B-Vision.md#run-llama-32-90b-vision-on-tt-loudbox) | - | - |
| [Qwen2.5-VL-72B-Instruct](vlm/Qwen2.5-VL-72B-Instruct.md) | - | - | [🟡 Functional](vlm/Qwen2.5-VL-72B-Instruct.md#run-qwen25-vl-72b-instruct-on-tt-loudbox) | - | - |

## Experimental Models

Models with EXPERIMENTAL status are under active development and may have stability or performance issues.

| Model Name | N150 | N300 | T3K | GALAXY | GALAXY_T3K |
| --- | --- | --- | --- | --- | --- |
| [gemma-3-27b-it](vlm/gemma-3-27b-it.md) | - | - | [🛠️ Experimental](vlm/gemma-3-27b-it.md#run-gemma-3-27b-it-on-tt-loudbox) | [🛠️ Experimental](vlm/gemma-3-27b-it.md#run-gemma-3-27b-it-on-tenstorrent-galaxy) | [🛠️ Experimental](vlm/gemma-3-27b-it.md#run-gemma-3-27b-it-on-tenstorrent-galaxy-galaxy_t3k) |
| [gemma-3-4b-it](vlm/gemma-3-4b-it.md) | [🛠️ Experimental](vlm/gemma-3-4b-it.md#run-gemma-3-4b-it-on-n150) | [🛠️ Experimental](vlm/gemma-3-4b-it.md#run-gemma-3-4b-it-on-n300) | - | - | - |
| [Qwen2.5-VL-32B-Instruct](vlm/Qwen2.5-VL-32B-Instruct.md) | - | - | [🛠️ Experimental](vlm/Qwen2.5-VL-32B-Instruct.md#run-qwen25-vl-32b-instruct-on-tt-loudbox) | - | - |
| [Qwen2.5-VL-3B-Instruct](vlm/Qwen2.5-VL-3B-Instruct.md) | [🛠️ Experimental](vlm/Qwen2.5-VL-3B-Instruct.md#run-qwen25-vl-3b-instruct-on-n150) | [🛠️ Experimental](vlm/Qwen2.5-VL-3B-Instruct.md#run-qwen25-vl-3b-instruct-on-n300) | [🛠️ Experimental](vlm/Qwen2.5-VL-3B-Instruct.md#run-qwen25-vl-3b-instruct-on-tt-loudbox) | - | - |
| [Qwen2.5-VL-7B-Instruct](vlm/Qwen2.5-VL-7B-Instruct.md) | [🛠️ Experimental](vlm/Qwen2.5-VL-7B-Instruct.md#run-qwen25-vl-7b-instruct-on-n150) | [🛠️ Experimental](vlm/Qwen2.5-VL-7B-Instruct.md#run-qwen25-vl-7b-instruct-on-n300) | [🛠️ Experimental](vlm/Qwen2.5-VL-7B-Instruct.md#run-qwen25-vl-7b-instruct-on-tt-loudbox) | - | - |
