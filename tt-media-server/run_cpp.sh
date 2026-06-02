# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# Start the C++ server
cd cpp_server

if [ "$LLM_DEVICE_BACKEND" = "pipeline_manager" ]; then
  LD_PRELOAD=$(pwd)/tt-llm-engine/build-full/libtt_llm_engine.so.0 ./build/tt_media_server_cpp -p 8000
else
  ./build/tt_media_server_cpp -p 8000
fi
