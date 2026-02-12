# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

cmake_minimum_required(VERSION 3.16)
project(tt_media_server_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Required dependency: tt-metal (TT-NN). Submodule in deps/tt-metal, install into deps/tt-metal/install.
# Override: set env TT_METAL_INSTALL_DIR or pass -DTT_METAL_INSTALL_DIR=/path.
set(TT_METAL_INSTALL_DIR "" CACHE PATH "Install prefix of tt-metal; default: deps/tt-metal/install")
if(NOT TT_METAL_INSTALL_DIR AND DEFINED ENV{TT_METAL_INSTALL_DIR})
    set(TT_METAL_INSTALL_DIR "$ENV{TT_METAL_INSTALL_DIR}")
endif()
if(NOT TT_METAL_INSTALL_DIR)
    set(TT_METAL_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/tt-metal/install")
endif()
if(NOT EXISTS "${TT_METAL_INSTALL_DIR}")
    message(FATAL_ERROR "tt-metal install not found at ${TT_METAL_INSTALL_DIR}. "
        "Build and install tt-metal (in deps/tt-metal: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && cmake --install build --prefix install)")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${TT_METAL_INSTALL_DIR}")
find_package(TT-NN REQUIRED)
message(STATUS "Using tt-metal (TT-NN) from ${TT_METAL_INSTALL_DIR}")

# Find Drogon
find_package(Drogon REQUIRED)

# Test build (used by PR gate). Pass -DTEST=ON or use build.sh --test.
option(TEST "Build for PR gate: LLM only, no embedding (no Python required)" OFF)
if(NOT TEST)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
endif()

# RapidJSON for fast streaming serialization (header-only; populate only to avoid RapidJSON's old cmake_minimum_required)
include(FetchContent)
FetchContent_Declare(
    rapidjson
    GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
    GIT_TAG        v1.1.0
)
# Conditionally set CMP0169 policy if supported
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()
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
target_link_libraries(llm_engine PUBLIC Boost::boost)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(llm_engine PUBLIC rt)
endif()
target_compile_features(llm_engine PUBLIC cxx_std_17)

option(LLM_ENGINE_DEBUG_BUILD "Enable LLM engine debug logging (LLM_ENGINE_LOG)" OFF)
if(LLM_ENGINE_DEBUG_BUILD)
    target_compile_definitions(llm_engine PUBLIC LLM_ENGINE_DEBUG)
endif()

option(SANITIZE_THREAD "Build with ThreadSanitizer (TSan) for data-race detection" OFF)
option(SANITIZE_ADDRESS "Build with AddressSanitizer (ASan) and LeakSanitizer for memory errors and leak detection" OFF)
if(SANITIZE_THREAD AND SANITIZE_ADDRESS)
    message(FATAL_ERROR "SANITIZE_THREAD and SANITIZE_ADDRESS are mutually exclusive; enable only one.")
endif()
if(SANITIZE_THREAD)
    target_compile_options(llm_engine PUBLIC -fsanitize=thread -g)
    target_link_options(llm_engine PUBLIC -fsanitize=thread)
endif()
if(SANITIZE_ADDRESS)
    set(ASAN_FLAGS -fsanitize=address -fno-omit-frame-pointer -g)
    target_compile_options(llm_engine PUBLIC ${ASAN_FLAGS})
    target_link_options(llm_engine PUBLIC -fsanitize=address)
endif()

add_executable(engine_demo src/runners/llm_engine/demo.cpp)
target_link_libraries(engine_demo PRIVATE llm_engine)
target_include_directories(engine_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/runners)

# Unit tests (LLM engine)
enable_testing()
add_executable(scheduler_test tests/scheduler_test.cpp)
target_link_libraries(scheduler_test PRIVATE llm_engine GTest::gtest_main)
target_include_directories(scheduler_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(llm_engine_test tests/llm_engine_test.cpp)
target_link_libraries(llm_engine_test PRIVATE llm_engine GTest::gtest_main)
target_include_directories(llm_engine_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(sequence_test tests/sequence_test.cpp)
target_link_libraries(sequence_test PRIVATE llm_engine GTest::gtest_main)
target_include_directories(sequence_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(scheduler_test)
gtest_discover_tests(llm_engine_test)
gtest_discover_tests(sequence_test)

# IPC smoke test (2-process, Boost.Interprocess queue + scheduler)
add_executable(ipc_scheduler_smoke_test tests/ipc_scheduler_smoke_test.cpp)
target_link_libraries(ipc_scheduler_smoke_test PRIVATE llm_engine)
target_include_directories(ipc_scheduler_smoke_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_test(NAME IpcSchedulerSmoke COMMAND ipc_scheduler_smoke_test)

# Source files
set(SOURCES
    src/main.cpp
    src/api/llm_controller.cpp
    src/util/mock_tokenizer.cpp
    src/api/openapi.cpp
    src/config/settings.cpp
    src/filters/security_filter.cpp
    src/scheduler/multiprocess_scheduler.cpp
    src/services/base_service.cpp
)
if(NOT TEST)
    list(APPEND SOURCES
        src/api/embedding_controller.cpp
        src/runners/embedding_runner.cpp
        src/services/embedding_service.cpp
    )
endif()

# Conditionally include embedding_runner.cpp only if ENABLE_TTNN is ON
if(ENABLE_TTNN)
    list(APPEND SOURCES src/runners/embedding_runner.cpp)
endif()

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${rapidjson_SOURCE_DIR}/include
)

# Link Drogon and llm_engine (used by multiprocess_scheduler)
target_link_libraries(${PROJECT_NAME} PRIVATE Drogon::Drogon llm_engine)
if(TEST)
    target_compile_definitions(${PROJECT_NAME} PRIVATE TEST)
endif()
if(NOT TEST)
    target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
endif()

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

# AddressSanitizer for executables (must be after all add_executable calls)
if(SANITIZE_ADDRESS)
    set(ASAN_FLAGS -fsanitize=address -fno-omit-frame-pointer -g)
    target_compile_options(${PROJECT_NAME} PRIVATE ${ASAN_FLAGS})
    target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
    target_compile_options(engine_demo PRIVATE ${ASAN_FLAGS})
    target_link_options(engine_demo PRIVATE -fsanitize=address)
    target_compile_options(scheduler_test PRIVATE ${ASAN_FLAGS})
    target_link_options(scheduler_test PRIVATE -fsanitize=address)
    target_compile_options(llm_engine_test PRIVATE ${ASAN_FLAGS})
    target_link_options(llm_engine_test PRIVATE -fsanitize=address)
    target_compile_options(sequence_test PRIVATE ${ASAN_FLAGS})
    target_link_options(sequence_test PRIVATE -fsanitize=address)
    target_compile_options(ipc_scheduler_smoke_test PRIVATE ${ASAN_FLAGS})
    target_link_options(ipc_scheduler_smoke_test PRIVATE -fsanitize=address)
endif()
