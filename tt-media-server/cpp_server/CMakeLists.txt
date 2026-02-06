# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

cmake_minimum_required(VERSION 3.16)
project(tt_media_server_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Drogon
find_package(Drogon REQUIRED)

# Python required for embedding runner (Python C API)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# RapidJSON for fast streaming serialization (header-only; populate only to avoid RapidJSON's old cmake_minimum_required)
include(FetchContent)
FetchContent_Declare(
    rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
    GIT_TAG        v1.1.0
)
cmake_policy(SET CMP0169 OLD)  # allow FetchContent_Populate (RapidJSON's CMake is too old for MakeAvailable)
FetchContent_Populate(rapidjson)

# Option for TTNN support
option(ENABLE_TTNN "Enable TTNN test runner" OFF)

# Nano vLLM engine (include/runners/nanovllm/, src/runners/nanovllm/)
set(NANOVLLM_ENGINE_SOURCES
    src/runners/nanovllm/engine/block_manager.cpp
    src/runners/nanovllm/engine/hash.cpp
    src/runners/nanovllm/engine/llm_engine.cpp
    src/runners/nanovllm/engine/model_runner.cpp
    src/runners/nanovllm/engine/scheduler.cpp
    src/runners/nanovllm/engine/sequence.cpp
)
add_library(nanovllm_engine ${NANOVLLM_ENGINE_SOURCES})
target_include_directories(nanovllm_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(nanovllm_engine PUBLIC NANOVLLM_USE_EXTERNAL_LOG)
target_compile_features(nanovllm_engine PUBLIC cxx_std_17)

add_executable(engine_demo src/runners/nanovllm/demo.cpp)
target_link_libraries(engine_demo PRIVATE nanovllm_engine)
target_include_directories(engine_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/runners)

add_executable(nanovllm_demo src/nanovllm_demo.cpp)
target_link_libraries(nanovllm_demo PRIVATE nanovllm_engine)
target_include_directories(nanovllm_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(nanovllm_demo PRIVATE -Wall -Wextra -Wpedantic)

# Source files
set(SOURCES
    src/main.cpp
    src/api/embedding_controller.cpp
    src/api/llm_controller.cpp
    src/api/openapi.cpp
    src/runners/embedding_runner.cpp
    src/scheduler/multiprocess_scheduler.cpp
    src/services/base_service.cpp
    src/services/embedding_service.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${rapidjson_SOURCE_DIR}/include
)

# Link Drogon and Python (embedding runner uses Python C API)
target_link_libraries(${PROJECT_NAME} PRIVATE Drogon::Drogon Python3::Python)

# TTNN support (optional)
if(ENABLE_TTNN)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_TTNN)
endif()

# Compiler warnings
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
endif()
