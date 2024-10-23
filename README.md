# TT-INFERENCE-SERVER

Tenstorrent Inference Server (TT-inference-server) is the repo of available model APIs for TT-studio

## Official Repository

[https://github.com/tenstorrent/tt-inference-server](https://github.com/tenstorrent/tt-inference-server/)

[https://github.com/tenstorrent/tt-studio](https://github.com/tenstorrent/tt-studio/)

## Getting started
Build and editing instruction are as follows -

## Download Model Weights  

Follow the instructions in TT Inference/**model to download the model weights.
Locate Model Weights  

The model weights should be in:  

**_persistent_volume/volume_id_tt-metal-***model_name***

Copy Weights to TT Studio  
Copy the weights into the TT Studio persistent directory. 
Run Docker mount access. 

Execute the instructions from TT Inference to ensure the folder has the required user and Docker mount access.

--------------------------------------------------------------------------------------------------------------

## Model implementations
| Model          | Hardware                    |
|----------------|-----------------------------|
| [LLaMa 3.1 70B](tt-metal-llama3-70b/README.md)  | TT-QuietBox & TT-LoudBox    |
| [Mistral 7B](tt-metal-mistral-7b/README.md) | n150 and n300|