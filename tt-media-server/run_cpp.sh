# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# Start the C++ server.
#
# Note on LD_PRELOAD removal (engine-split Phase 3c):
# The previous version preloaded ./tt-llm-engine/build-full/libtt_llm_engine.so.0
# when LLM_DEVICE_BACKEND=pipeline_manager. That path no longer exists — the
# engine .so now lives at /opt/tt-llm-engine/lib/libtt_llm_engine.so.0 (shipped
# by the prebuilt tt-llm-engine image) and is linked into tt_media_server_cpp
# via find_package (cpp_server/CMakeLists.txt) plus runtime LD_LIBRARY_PATH
# (set in Dockerfile.blaze runtime stage). There is now a single canonical
# copy of the .so, so the preload override that distinguished build-tree from
# install-tree copies is no longer needed.

cd cpp_server
if [ "${MODEL:-${HF_MODEL_ID:-}}" = "meta-llama/Llama-3.2-1B-Instruct" ]; then
    # Generic images omit these optional gated assets. Fetch only the selected
    # model before starting the server, or reuse a complete offline cache.
    export MODEL="meta-llama/Llama-3.2-1B-Instruct"
    ./scripts/fetch_tokenizers.sh ./tokenizers "${MODEL}" || exit 1
fi
./build/tt_media_server_cpp -p 8000
