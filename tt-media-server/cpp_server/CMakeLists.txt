# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

cmake_minimum_required(VERSION 3.19)
project(tt_media_server_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Allow config files (e.g. system jsoncpp) that use older cmake_minimum_required (CMake 4.x compatibility)
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Min policy version for included configs")
endif()

# Optional clang-tidy during build (same approach as tt-metal: lint = build with clang-tidy)
# Set only after FetchContent so _deps (tokenizers_cpp, sentencepiece, protobuf, etc.) are excluded.
option(CLANG_TIDY "Run clang-tidy during build" OFF)

# Find Drogon and JsonCpp (tokenizer_config parsing)
find_package(Drogon REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Try to find cereal, or fetch it if not available
find_package(cereal QUIET)
if(NOT cereal_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        cereal
        GIT_REPOSITORY https://github.com/USCiLab/cereal.git
        GIT_TAG v1.3.2
    )

    # Disable cereal tests and examples
    set(BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_SANDBOX OFF CACHE BOOL "" FORCE)
    set(SKIP_PERFORMANCE_COMPARISON ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(cereal)
endif()

# RapidJSON for fast streaming serialization (header-only; populate only to avoid RapidJSON's old cmake_minimum_required)
include(FetchContent)

# pybind11 for embedding Python interpreter (in-process model runner calls)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)
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

# Import fmt from tt-metal for spdlog
if(TT_METAL_HOME AND EXISTS "${TT_METAL_HOME}/build_Release/lib/cmake/fmt")
    set(fmt_DIR "${TT_METAL_HOME}/build_Release/lib/cmake/fmt")
    find_package(fmt QUIET)
    if(fmt_FOUND)
        message(STATUS "Found fmt from tt-metal: ${fmt_VERSION}")
        set(USE_EXTERNAL_FMT ON)
    else()
        message(STATUS "fmt directory found but package not available, using spdlog's bundled fmt")
        set(USE_EXTERNAL_FMT OFF)
    endif()
else()
    set(USE_EXTERNAL_FMT OFF)
    message(STATUS "Using spdlog's bundled fmt library")
endif()

# Ensure static libraries are built with -fPIC (required when linking into shared libs like pipeline_manager)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# spdlog for logging - use tt-metal's fmt library when available
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.17.0
)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ${USE_EXTERNAL_FMT} CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spdlog)

# Boost (header-only Boost.Interprocess for IPC message queue)
find_package(Boost REQUIRED)

# tt-blaze pipeline_manager
# Make TT-Metalium discoverable so pipeline_manager builds the Full target
# (SocketPipeline) instead of core-only (mock pipeline).
if(TT_METAL_HOME)
    foreach(_build_dir build_Release build)
        if(EXISTS "${TT_METAL_HOME}/${_build_dir}/lib/cmake/tt-metalium")
            list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_HOME}/${_build_dir}")
            break()
        endif()
    endforeach()
endif()
set(PM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PM_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(TT_METAL_SOURCE_DIR "${TT_METAL_HOME}" CACHE PATH "" FORCE)
add_subdirectory(tt-blaze/pipeline_manager)

# Option for Tracy support
option(ENABLE_TRACY "Enable Tracy profiling instrumentation" OFF)

if(ENABLE_TRACY)
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/tenstorrent/tracy.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(tracy)

    add_library(tracy_client STATIC ${tracy_SOURCE_DIR}/public/TracyClient.cpp)
    target_include_directories(tracy_client PUBLIC ${tracy_SOURCE_DIR}/public)
    target_compile_definitions(tracy_client PUBLIC
        TRACY_ENABLE
        TRACY_DELAYED_INIT
        TRACY_MANUAL_LIFETIME
    )
endif()

# Prometheus metrics (core only — no embedded HTTP server, no push gateway)
set(ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(ENABLE_PUSH OFF CACHE BOOL "" FORCE)
set(ENABLE_PULL OFF CACHE BOOL "" FORCE)
set(ENABLE_COMPRESSION OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    prometheus-cpp
    GIT_REPOSITORY https://github.com/jupp0r/prometheus-cpp.git
    GIT_TAG        v1.2.4
    GIT_SHALLOW    ON
    GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(prometheus-cpp)

# Tokenizer (mlc-ai/tokenizers-cpp). Requires Rust.
set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        55d53aa38dc8df7d9c8bd9ed50907e82ae83ce66
    GIT_SUBMODULES "msgpack" "sentencepiece"
)
FetchContent_MakeAvailable(tokenizers_cpp)

# Enable clang-tidy only for our code (not _deps). Must be after FetchContent.
if(CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy-20 clang-tidy-19 clang-tidy)
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE};--warnings-as-errors=*")
  else()
    message(FATAL_ERROR "CLANG_TIDY=ON but no clang-tidy found. Install clang-tidy-20 or run install_dependencies.sh")
  endif()
endif()

# TT Metal C++ API: headers from TT_METAL_HOME (for referencing Metal API, intellisense)
if(NOT DEFINED TT_METAL_HOME AND DEFINED ENV{TT_METAL_HOME})
    set(TT_METAL_HOME "$ENV{TT_METAL_HOME}" CACHE PATH "TT Metal home (from TT_METAL_HOME env)")
endif()
if(TT_METAL_HOME AND EXISTS "${TT_METAL_HOME}/tt_metal/api")
    set(TT_METAL_API_FOUND ON)
    set(TT_METAL_INCLUDE_DIRS
        "${TT_METAL_HOME}"
        "${TT_METAL_HOME}/tt_metal"
        "${TT_METAL_HOME}/tt_metal/api"
        "${TT_METAL_HOME}/tt_metal/third_party/umd"
        "${TT_METAL_HOME}/tt_metal/third_party/umd/device/api"
        "${TT_METAL_HOME}/tt_metal/hostdevcommon/api"
        "${TT_METAL_HOME}/tt_metal/hw/inc"
        "${TT_METAL_HOME}/tt_metal/hw/inc/internal/tt-1xx/wormhole"
        "${TT_METAL_HOME}/tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines"
        "${TT_METAL_HOME}/tt_metal/hw/inc/internal/tt-1xx/blackhole"
        "${TT_METAL_HOME}/tt_metal/third_party/tracy/public"
        "${TT_METAL_HOME}/tt_stl"
    )
    # Include build directories
    if(EXISTS "${TT_METAL_HOME}/build_Release/include")
        list(APPEND TT_METAL_INCLUDE_DIRS
            "${TT_METAL_HOME}/build_Release/include"
            "${TT_METAL_HOME}/build_Release/include/metalium-thirdparty")
    elseif(EXISTS "${TT_METAL_HOME}/build/include")
        list(APPEND TT_METAL_INCLUDE_DIRS
            "${TT_METAL_HOME}/build/include"
            "${TT_METAL_HOME}/build/include/metalium-thirdparty")
    elseif(EXISTS "${TT_METAL_HOME}/build/_deps/enchantum-src/include")
        list(APPEND TT_METAL_INCLUDE_DIRS "${TT_METAL_HOME}/build/_deps/enchantum-src/include")
    elseif(EXISTS "${TT_METAL_HOME}/.cpmcache/enchantum")
        file(GLOB _enchantum_dirs "${TT_METAL_HOME}/.cpmcache/enchantum/*/enchantum/include")
        if(_enchantum_dirs)
            list(GET _enchantum_dirs 0 _enchantum_include)
            list(APPEND TT_METAL_INCLUDE_DIRS "${_enchantum_include}")
        endif()
    endif()
    message(STATUS "TT Metal C++ API: ${TT_METAL_HOME} (includes + intellisense)")
else()
    set(TT_METAL_API_FOUND OFF)
    set(TT_METAL_INCLUDE_DIRS "")
    if(TT_METAL_HOME)
        message(STATUS "TT Metal C++ API: TT_METAL_HOME set but tt_metal/api not found, skipping")
    endif()
endif()

# LLM engine (include/runners/llm_runner/, src/runners/llm_runner/)
set(LLM_ENGINE_SOURCES
    src/runners/llm_runner/block_manager.cpp
    src/ipc/boost_ipc_cancel_queue.cpp
    src/ipc/boost_ipc_task_queue.cpp
    src/runners/llm_runner/hash.cpp
    src/runners/llm_runner.cpp
    src/runners/llm_runner/model_runner.cpp
    src/runners/llm_runner/mock_model_runner.cpp
    src/runners/llm_runner/scheduler.cpp
    src/runners/llm_runner/sequence.cpp
    src/runners/llm_runner/sampling_params.cpp
    src/utils/logger.cpp
    src/config/settings.cpp
    src/utils/tokenizer.cpp
    src/utils/deepseek_tokenizer.cpp
    src/utils/llama_tokenizer.cpp
    src/utils/tokenizer_config_loader.cpp
    src/runners/sp_pipeline_runner/sp_pipeline_runner.cpp
    src/runners/sp_pipeline_runner/sp_pipeline_runner_demo.cpp
    src/runners/sp_pipeline_runner/i_sp_pipeline_model_runner.cpp
    src/runners/sp_pipeline_runner/sp_pipeline_model_runner.cpp
    src/runners/sp_pipeline_runner/mock_device_pipeline.cpp
    src/runners/sp_pipeline_runner/mock_sp_pipeline_model_runner.cpp
    src/runners/sp_prefill_runner/sp_prefill_runner.cpp
    src/runners/sp_prefill_runner/i_sp_prefill_model_runner.cpp
    src/runners/sp_prefill_runner/sp_prefill_model_runner.cpp
    src/services/memory_manager.cpp
    src/services/paged_memory_manager.cpp
    src/services/contiguous_memory_manager.cpp
    src/services/sp_pipeline_memory_manager.cpp
)
# Add socket backend when tt-metal is available
if(TT_METAL_API_FOUND AND TT_METAL_INCLUDE_DIRS)
    list(APPEND LLM_ENGINE_SOURCES
        src/runners/llama_model_runner.cpp
    )
    message(STATUS "LLM engine: building with tt-metal socket integration")
else()
    message(STATUS "LLM engine: building without tt-metal (tt-metal not found)")
endif()
add_library(llm_engine ${LLM_ENGINE_SOURCES})
target_include_directories(llm_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TT_METAL_INCLUDE_DIRS}
)
# Link tokenizers_cpp and JsonCpp since settings.cpp depends on tokenizer.hpp
# Use PipelineManager::Full (with SocketPipeline + TT::Metalium) when available,
# otherwise fall back to PipelineManager::Core (mock pipeline only).
if(TARGET PipelineManager::Full)
    target_link_libraries(llm_engine PUBLIC tokenizers_cpp JsonCpp::JsonCpp PipelineManager::Full)
else()
    target_link_libraries(llm_engine PUBLIC tokenizers_cpp JsonCpp::JsonCpp PipelineManager::Core)
endif()
if(TT_METAL_API_FOUND AND TT_METAL_INCLUDE_DIRS)
    target_compile_definitions(llm_engine PUBLIC USE_METAL_CPP_LIB)
    # Find tt_metal library directory
    set(_tt_metal_lib_dir "")
    if(EXISTS "${TT_METAL_HOME}/build_Release/lib/libtt_metal.so")
        set(_tt_metal_lib_dir "${TT_METAL_HOME}/build_Release/lib")
    elseif(EXISTS "${TT_METAL_HOME}/build/lib/libtt_metal.so")
        set(_tt_metal_lib_dir "${TT_METAL_HOME}/build/lib")
    endif()
    if(_tt_metal_lib_dir)
        target_link_directories(llm_engine PUBLIC "${_tt_metal_lib_dir}")
        target_link_libraries(llm_engine PUBLIC tt_metal tt_stl fmt)
    endif()
    target_link_libraries(llm_engine PUBLIC pybind11::embed)
endif()
if(USE_EXTERNAL_FMT)
    target_link_libraries(llm_engine PUBLIC Boost::boost spdlog::spdlog fmt::fmt)
else()
    target_link_libraries(llm_engine PUBLIC Boost::boost spdlog::spdlog)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(llm_engine PUBLIC rt)
endif()
target_compile_features(llm_engine PUBLIC cxx_std_20)

if(ENABLE_TRACY)
    target_link_libraries(llm_engine PUBLIC tracy_client)
endif()

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

# Unit tests (LLM engine)
enable_testing()
add_executable(scheduler_test tests/scheduler_test.cpp)
target_link_libraries(scheduler_test PRIVATE llm_engine GTest::gtest_main)
target_include_directories(scheduler_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(llm_runner_test tests/llm_runner_test.cpp)
target_link_libraries(llm_runner_test PRIVATE llm_engine GTest::gtest_main)
target_include_directories(llm_runner_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(sequence_test tests/sequence_test.cpp)
target_link_libraries(sequence_test PRIVATE llm_engine GTest::gtest_main)
target_include_directories(sequence_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(lockfree_queue_test tests/lockfree_queue_test.cpp)
target_link_libraries(lockfree_queue_test PRIVATE GTest::gtest_main)
target_include_directories(lockfree_queue_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Tokenizer test
add_executable(test_tokenizer tests/test_tokenizer.cpp src/utils/tokenizer.cpp src/utils/deepseek_tokenizer.cpp src/utils/llama_tokenizer.cpp src/utils/tokenizer_config_loader.cpp src/config/settings.cpp src/utils/logger.cpp)
if(USE_EXTERNAL_FMT)
    target_link_libraries(test_tokenizer PRIVATE GTest::gtest_main tokenizers_cpp JsonCpp::JsonCpp spdlog::spdlog fmt::fmt)
else()
    target_link_libraries(test_tokenizer PRIVATE GTest::gtest_main tokenizers_cpp JsonCpp::JsonCpp spdlog::spdlog)
endif()
target_include_directories(test_tokenizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(cancel_queue_test tests/cancel_queue_test.cpp)
target_link_libraries(cancel_queue_test PRIVATE llm_engine GTest::gtest_main)
target_include_directories(cancel_queue_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(scheduler_test)
gtest_discover_tests(llm_runner_test)
gtest_discover_tests(sequence_test)
gtest_discover_tests(lockfree_queue_test)
gtest_discover_tests(test_tokenizer)
gtest_discover_tests(cancel_queue_test)

# IPC smoke test (2-process, Boost.Interprocess queue + scheduler)
add_executable(ipc_scheduler_smoke_test tests/ipc_scheduler_smoke_test.cpp)
target_link_libraries(ipc_scheduler_smoke_test PRIVATE llm_engine)
target_include_directories(ipc_scheduler_smoke_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_test(NAME IpcSchedulerSmoke COMMAND ipc_scheduler_smoke_test)

# Tokenizer benchmark
add_executable(tokenizer_benchmark benchmarks/tokenizer_benchmark.cpp src/utils/tokenizer.cpp src/utils/deepseek_tokenizer.cpp src/utils/llama_tokenizer.cpp src/utils/tokenizer_config_loader.cpp src/config/settings.cpp src/utils/logger.cpp)
target_include_directories(tokenizer_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(USE_EXTERNAL_FMT)
    target_link_libraries(tokenizer_benchmark PRIVATE tokenizers_cpp JsonCpp::JsonCpp spdlog::spdlog fmt::fmt)
else()
    target_link_libraries(tokenizer_benchmark PRIVATE tokenizers_cpp JsonCpp::JsonCpp spdlog::spdlog)
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/profiling/tracy.cpp
    src/api/llm_controller.cpp
    src/api/health_controller.cpp
    src/api/openapi.cpp
    src/config/settings.cpp
    src/ipc/boost_ipc_warmup_signal_queue.cpp
    src/filters/security_filter.cpp
    src/services/llm_service.cpp
    src/services/memory_manager.cpp
    src/services/paged_memory_manager.cpp
    src/services/contiguous_memory_manager.cpp
    src/services/reasoning_parser.cpp
    src/services/session_manager.cpp
    src/domain/session.cpp
    src/utils/tokenizer.cpp
    src/utils/logger.cpp
    src/utils/deepseek_tokenizer.cpp
    src/utils/llama_tokenizer.cpp
    src/worker/single_process_worker.cpp
    src/worker/worker_manager.cpp
    src/utils/service_factory.cpp
    src/utils/service_container.cpp
    src/utils/runner_factory.cpp
    src/utils/tokenizer_config_loader.cpp
    src/utils/mapper.cpp
    src/api/embedding_controller.cpp
    src/runners/embedding_runner.cpp
    src/services/embedding_service.cpp
    src/services/disaggregation_service.cpp
    src/sockets/socket_manager.cpp
    src/sockets/inter_server_service.cpp
    src/metrics/metrics.cpp
    src/api/metrics_controller.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/runners
    ${rapidjson_SOURCE_DIR}/include
    ${TT_METAL_INCLUDE_DIRS}
    ${prometheus-cpp_SOURCE_DIR}/core/include
)

# Compile definitions for performance optimization
target_compile_definitions(${PROJECT_NAME} PRIVATE)

# Link Drogon, llm_engine, and prometheus-cpp core
if(USE_EXTERNAL_FMT)
    target_link_libraries(${PROJECT_NAME} PRIVATE Drogon::Drogon llm_engine cereal::cereal spdlog::spdlog fmt::fmt prometheus-cpp::core)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE Drogon::Drogon llm_engine cereal::cereal spdlog::spdlog prometheus-cpp::core)
endif()
if(ENABLE_TRACY)
    target_sources(${PROJECT_NAME} PRIVATE src/profiling/tracy_alloc.cpp)
    target_compile_definitions(${PROJECT_NAME} PRIVATE TT_ENABLE_TRACY)
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::embed)

# Tokenizer (requires Rust) and tokenizer_config (JsonCpp)
target_link_libraries(${PROJECT_NAME} PRIVATE tokenizers_cpp JsonCpp::JsonCpp)

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
    target_compile_options(scheduler_test PRIVATE ${ASAN_FLAGS})
    target_link_options(scheduler_test PRIVATE -fsanitize=address)
    target_compile_options(llm_runner_test PRIVATE ${ASAN_FLAGS})
    target_link_options(llm_runner_test PRIVATE -fsanitize=address)
    target_compile_options(sequence_test PRIVATE ${ASAN_FLAGS})
    target_link_options(sequence_test PRIVATE -fsanitize=address)
    target_compile_options(ipc_scheduler_smoke_test PRIVATE ${ASAN_FLAGS})
    target_link_options(ipc_scheduler_smoke_test PRIVATE -fsanitize=address)
    target_compile_options(lockfree_queue_test PRIVATE ${ASAN_FLAGS})
    target_link_options(lockfree_queue_test PRIVATE -fsanitize=address)
    target_compile_options(cancel_queue_test PRIVATE ${ASAN_FLAGS})
    target_link_options(cancel_queue_test PRIVATE -fsanitize=address)
endif()
