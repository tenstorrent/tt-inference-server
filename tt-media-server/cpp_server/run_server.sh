#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# Navigate to the cpp_server directory
cd "$(dirname "$0")"

# Set up runtime environment variables
export TT_METAL_HOME=/data/TT_USER/tt-inference-server/tt-media-server/cpp_server/tt-llm-engine/tt-metal
export LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib:$(pwd)/tt-llm-engine/build-full:$LD_LIBRARY_PATH

# Run the server
LLM_DEVICE_BACKEND=pipeline_manager \
DEVICE_IDS=0,1,4,5,24,25,28,29 \
LD_PRELOAD=$(pwd)/tt-llm-engine/build-full/libtt_llm_engine.so.0 \
./build/tt_media_server_cpp
