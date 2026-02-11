# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# Optional third-party dependencies: fetched at configure time via FetchContent
# (no git submodules). Fetched content goes to build/_deps/<name>-src.
#
# Add new optional deps here: check option, FetchContent_Declare + MakeAvailable.

if(ENABLE_TOKENIZER)
    set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER ON CACHE BOOL "" FORCE)
    include(FetchContent)
    FetchContent_Declare(
        tokenizers_cpp
        GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
        GIT_TAG        e8964871dda750164787d8fba23e1451918677d4
        GIT_SUBMODULES "msgpack" "sentencepiece"
    )
    FetchContent_MakeAvailable(tokenizers_cpp)
endif()
