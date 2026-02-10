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

# Google Test for unit tests
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Boost (header-only Boost.Interprocess for IPC message queue)
find_package(Boost REQUIRED)

# Option for TTNN support
option(ENABLE_TTNN "Enable TTNN test runner" OFF)

# LLM engine (include/runners/llm_engine/, src/runners/llm_engine/)
set(LLM_ENGINE_SOURCES
    src/runners/llm_engine/engine/block_manager.cpp
    src/runners/llm_engine/engine/boost_ipc_task_queue.cpp
    src/runners/llm_engine/engine/hash.cpp
    src/runners/llm_engine/engine/llm_engine.cpp
    src/runners/llm_engine/engine/model_runner.cpp
    src/runners/llm_engine/engine/scheduler.cpp
    src/runners/llm_engine/engine/sequence.cpp
)
add_library(llm_engine ${LLM_ENGINE_SOURCES})
target_include_directories(llm_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(llm_engine PUBLIC Boost::boost rt)
target_compile_definitions(llm_engine PUBLIC LLM_ENGINE_USE_EXTERNAL_LOG)
target_compile_features(llm_engine PUBLIC cxx_std_17)

add_executable(engine_demo src/runners/llm_engine/demo.cpp)
target_link_libraries(engine_demo PRIVATE llm_engine)
target_include_directories(engine_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/runners)

add_executable(llm_engine_demo src/llm_engine_demo.cpp)
target_link_libraries(llm_engine_demo PRIVATE llm_engine)
target_include_directories(llm_engine_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(llm_engine_demo PRIVATE -Wall -Wextra -Wpedantic)

# Unit tests (LLM engine scheduler)
enable_testing()
add_executable(scheduler_test tests/scheduler_test.cpp)
target_link_libraries(scheduler_test PRIVATE llm_engine GTest::gtest_main)
target_include_directories(scheduler_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
include(GoogleTest)
gtest_discover_tests(scheduler_test)

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
