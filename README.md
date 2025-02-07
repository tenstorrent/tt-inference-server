# TT-Inference-Server

Tenstorrent Inference Server (`tt-inference-server`) is the repo of available model APIs for deploying on Tenstorrent hardware.

## Official Repository

[https://github.com/tenstorrent/tt-inference-server](https://github.com/tenstorrent/tt-inference-server/)


## Getting Started
Please follow setup instructions for the model you want to serve, `Model Name` in tables below link to corresponding implementation.

Note: models with Status [🔍 preview] are under active development. If you encounter setup or stability problems please [file an issue](https://github.com/tenstorrent/tt-inference-server/issues/new?template=Blank+issue) and our team will address it.

## LLMs

| Model Name                    | Model URL                                                             | Hardware                                                                 | Status      | Minimum Release Version                                                          |
| ----------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------- | -------------------------------------------------------------------------------- |
| [Qwen2.5-72B-Instruct](vllm-tt-metal-llama3/README.md)          | [HF Repo](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)           | [TT-QuietBox & TT-LoudBox](https://tenstorrent.com/hardware/tt-quietbox) | 🔍 preview  | [v0.0.2](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.2) |
| [Qwen2.5-72B](vllm-tt-metal-llama3/README.md)                   | [HF Repo](https://huggingface.co/Qwen/Qwen2.5-72B)                    | [TT-QuietBox & TT-LoudBox](https://tenstorrent.com/hardware/tt-quietbox) | 🔍 preview  | [v0.0.2](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.2) |
| [Qwen2.5-7B-Instruct](vllm-tt-metal-llama3/README.md)           | [HF Repo](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)            | [n150](https://tenstorrent.com/hardware/wormhole)                        | 🔍 preview  | [v0.0.2](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.2) |
| [Qwen2.5-7B](vllm-tt-metal-llama3/README.md)                    | [HF Repo](https://huggingface.co/Qwen/Qwen2.5-7B)                     | [n150](https://tenstorrent.com/hardware/wormhole)                        | 🔍 preview  | [v0.0.2](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.2) |
| [Llama-3.3-70B-Instruct](vllm-tt-metal-llama3/README.md)        | [HF Repo](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)        | [TT-QuietBox & TT-LoudBox](https://tenstorrent.com/hardware/tt-quietbox) | ✅ supported | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |
| [Llama-3.3-70B](vllm-tt-metal-llama3/README.md)                 | [HF Repo](https://huggingface.co/meta-llama/Llama-3.3-70B)                 | [TT-QuietBox & TT-LoudBox](https://tenstorrent.com/hardware/tt-quietbox) | ✅ supported | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |
| [Llama-3.2-11B-Vision-Instruct](vllm-tt-metal-llama3/README.md) | [HF Repo](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) | [n300](https://tenstorrent.com/hardware/wormhole)                        | 🔍 preview  | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |
| [Llama-3.2-11B-Vision](vllm-tt-metal-llama3/README.md)          | [HF Repo](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)          | [n300](https://tenstorrent.com/hardware/wormhole)                        | 🔍 preview  | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |
| [Llama-3.2-3B-Instruct](vllm-tt-metal-llama3/README.md)         | [HF Repo](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)         | [n150](https://tenstorrent.com/hardware/wormhole)                        | 🔍 preview  | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |
| [Llama-3.2-3B](vllm-tt-metal-llama3/README.md)                  | [HF Repo](https://huggingface.co/meta-llama/Llama-3.2-3B)                  | [n150](https://tenstorrent.com/hardware/wormhole)                        | 🔍 preview  | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |
| [Llama-3.2-1B-Instruct](vllm-tt-metal-llama3/README.md)         | [HF Repo](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)         | [n150](https://tenstorrent.com/hardware/wormhole)                        | 🔍 preview  | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |
| [Llama-3.2-1B](vllm-tt-metal-llama3/README.md)                  | [HF Repo](https://huggingface.co/meta-llama/Llama-3.2-1B)                  | [n150](https://tenstorrent.com/hardware/wormhole)                        | 🔍 preview  | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |
| [Llama-3.1-70B-Instruct](vllm-tt-metal-llama3/README.md)        | [HF Repo](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)        | [TT-QuietBox & TT-LoudBox](https://tenstorrent.com/hardware/tt-quietbox) | ✅ supported | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |
| [Llama-3.1-70B](vllm-tt-metal-llama3/README.md)                 | [HF Repo](https://huggingface.co/meta-llama/Llama-3.1-70B)                 | [TT-QuietBox & TT-LoudBox](https://tenstorrent.com/hardware/tt-quietbox) | ✅ supported | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |
| [Llama-3.1-8B-Instruct](vllm-tt-metal-llama3/README.md)         | [HF Repo](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)         | [n150](https://tenstorrent.com/hardware/wormhole)                        | ✅ supported | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |
| [Llama-3.1-8B](vllm-tt-metal-llama3/README.md)                  | [HF Repo](https://huggingface.co/meta-llama/Llama-3.1-8B)                  | [n150](https://tenstorrent.com/hardware/wormhole)                        | ✅ supported | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |

# CNNs

| Model Name                    | Model URL                                                             | Hardware                                                                 | Status      | Minimum Release Version                                                          |
| ----------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------- | -------------------------------------------------------------------------------- |
| [YOLOv4](tt-metal-yolov4/README.md)                        | [GH Repo](https://github.com/AlexeyAB/darknet)                    | [n150](https://tenstorrent.com/hardware/wormhole)                        | 🔍 preview  | [v0.0.1](https://github.com/tenstorrent/tt-inference-server/releases/tag/v0.0.1) |

