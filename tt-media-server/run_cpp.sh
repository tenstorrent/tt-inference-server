# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# Start the C++ server
cd cpp_server

if [ "$LLM_DEVICE_BACKEND" = "pipeline_manager" ]; then
  LD_PRELOAD=$(pwd)/tt-blaze/pipeline_manager/build-full/libpipeline_manager.so.1 ./build/tt_media_server_cpp -p 8000
else
  ./build/tt_media_server_cpp -p 8000
fi
