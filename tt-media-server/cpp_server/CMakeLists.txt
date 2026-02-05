# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

cmake_minimum_required(VERSION 3.16)
project(tt_media_server_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Drogon
find_package(Drogon REQUIRED)

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

# Link Drogon
target_link_libraries(${PROJECT_NAME} PRIVATE Drogon::Drogon)


# TTNN support (optional)
if(ENABLE_TTNN)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_TTNN)
    target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
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
