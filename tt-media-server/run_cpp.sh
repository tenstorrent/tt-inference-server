#!/bin/bash
# filepath: /localdev/idjuric/tt-inference-server/tt-media-server/run_cpp.sh

# Start the C++ server
cd cpp_server/build
./tt_media_server_cpp -p 8000
