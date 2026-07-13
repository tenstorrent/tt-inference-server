#!/bin/bash

cd ~/tt-xla && source venv/activate
cd ~/tt-inference-server/tt-media-server
# Install ONLY the server deps. Do NOT `pip install -r requirements.txt` into tt-xla venv.
# pip install aiohttp colorama fastapi faster_fifo httpx huggingface_hub loguru \
#       prometheus-client prometheus-fastapi-instrumentator psutil pydantic-settings \
#       python-multipart tqdm uvicorn requests num2words pytest pytest-asyncio

export MODEL="Mistral-Small-3.1-24B-Instruct-2503"
export DEVICE="bh-galaxy"
export MODEL_RUNNER="vllm_forge_mistral_small_31_24b"
export MAX_MODEL_LENGTH="8192" # should be as high as possible
export MAX_NUM_SEQS="32" # b32
export GPU_MEMORY_UTILIZATION="0.65" # ~33.15x concurrency
export CPU_SAMPLING="true" # at some point needs to be false
export OPTIMIZATION_LEVEL="1"
export ENABLE_TRACE="true"
export SERVICE_PORT="8019" # uvicorn http port
# set TT_METAL_HOME and TT_METAL_RUNTIME_ROOT to the correct value
export TT_METAL_HOME="/home/acicovic/tt-xla/python_package/pjrt_plugin_tt/tt-metal"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"
export TT_MESH_GRAPH_DESC_PATH="${TT_METAL_HOME}/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"

# --skip-venv: assume the tt-xla venv is already active
exec ./run_uvicorn.sh --skip-venv
