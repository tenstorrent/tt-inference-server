# TT-INFERENCE-SERVER

Tenstorrent Inference Server (`tt-inference-server`) is the repo of available model APIs for deploying on Tenstorrent hardware.

## Official Repository

[https://github.com/tenstorrent/tt-inference-server](https://github.com/tenstorrent/tt-inference-server/)


## Getting started
Build and editing instruction are as follows -


## Download Model Weights  

Follow the instructions in TT Inference/**model to download the model weights.

## Usage with a TT Project

Execute the instructions from TT Inference to ensure the folder has the required user and Docker mount access.

--------------------------------------------------------------------------------------------------------------

## Model implementations
| Model          | Hardware                    |
|----------------|-----------------------------|
| [LLaMa 3.1 70B](tt-metal-llama3-70b/README.md)  | TT-QuietBox & TT-LoudBox    |
| [Mistral 7B](tt-metal-mistral-7b/README.md) | n150 and n300|