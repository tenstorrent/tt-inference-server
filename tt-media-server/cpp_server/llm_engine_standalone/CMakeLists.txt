# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# Standalone build: LLM engine library + demo only (no Drogon/Python).
# Run from cpp_server. Use the same C++ compiler as tt-metal (e.g. clang++-20), e.g.:
#   cmake -B build -S llm_engine_standalone -DCMAKE_CXX_COMPILER=clang++-20
#   cmake --build build && ./build/llm_engine_demo

cmake_minimum_required(VERSION 3.16)
project(llm_engine_demo CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CPP_SERVER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(TT_METAL_BUILD_DIR "/data/dmadic/tt-metal/build_Release" CACHE PATH "Path to tt-metal build directory (e.g. build_Release)")

set(LLM_ENGINE_SOURCES
    ${CPP_SERVER_DIR}/src/runners/llm_engine/engine/block_manager.cpp
    ${CPP_SERVER_DIR}/src/runners/llm_engine/engine/device_context_ttmetal.cpp
    ${CPP_SERVER_DIR}/src/runners/llm_engine/engine/hash.cpp
    ${CPP_SERVER_DIR}/src/runners/llm_engine/engine/llm_engine.cpp
    ${CPP_SERVER_DIR}/src/runners/llm_engine/engine/model_runner.cpp
    ${CPP_SERVER_DIR}/src/runners/llm_engine/engine/scheduler.cpp
    ${CPP_SERVER_DIR}/src/runners/llm_engine/engine/sequence.cpp
)
add_library(llm_engine ${LLM_ENGINE_SOURCES})
target_include_directories(llm_engine PUBLIC
    ${CPP_SERVER_DIR}/include/runners
    ${CPP_SERVER_DIR}/include
)
target_compile_definitions(llm_engine PUBLIC LLM_ENGINE_DEBUG)
list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_BUILD_DIR}/lib/cmake" "${TT_METAL_BUILD_DIR}")
find_package(TT-Metalium REQUIRED)
get_filename_component(TT_METAL_SOURCE_DIR "${TT_METAL_BUILD_DIR}/.." ABSOLUTE)
target_compile_definitions(llm_engine PRIVATE "TT_METAL_RUNTIME_ROOT_DEFAULT=\"${TT_METAL_SOURCE_DIR}\"")
target_include_directories(llm_engine PRIVATE "${TT_METAL_SOURCE_DIR}/tt_metal/api")
target_link_libraries(llm_engine PRIVATE TT::Metalium)
target_compile_features(llm_engine PUBLIC cxx_std_20)

add_executable(llm_engine_demo ${CPP_SERVER_DIR}/src/llm_engine_demo.cpp)
target_link_libraries(llm_engine_demo PRIVATE llm_engine)
target_include_directories(llm_engine_demo PRIVATE
    ${CPP_SERVER_DIR}/include/runners
    ${CPP_SERVER_DIR}/include
)
target_compile_options(llm_engine_demo PRIVATE -Wall -Wextra -Wpedantic)
