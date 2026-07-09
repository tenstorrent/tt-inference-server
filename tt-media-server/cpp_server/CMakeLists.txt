# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

cmake_minimum_required(VERSION 3.24)
# C and ASM are enabled for the BLAKE3 C library (FetchContent below), whose
# optimized SIMD paths compile .c and .S sources. Enabling them on the
# top-level project ensures CMake detects the C/ASM toolchain up front instead
# of failing once BLAKE3's subdirectory pulls them in.
project(tt_media_server_cpp CXX C ASM)

# Mooncake's limit_jobs.cmake (included only when ENABLE_MOONCAKE=ON, via the
# add_subdirectory below) declares ninja "compile_pool"/"link_pool" through the
# JOB_POOLS global property AND sets CMAKE_JOB_POOL_COMPILE/LINK as cached FORCE
# vars. The cache vars persist across reconfigures but the (non-cached) JOB_POOLS
# property does not. Reusing a Mooncake-enabled build dir with ENABLE_MOONCAKE=OFF
# would otherwise leave every target referencing an undeclared pool ->
# "ninja: unknown pool name 'compile_pool'". Clear the leaked cache vars before any
# target is created (CMAKE_JOB_POOL_* is captured at target-creation time); when
# Mooncake is enabled, limit_jobs.cmake re-declares both pool and var in time for
# the heavy targets, which are all defined after the add_subdirectory.
unset(CMAKE_JOB_POOL_COMPILE CACHE)
unset(CMAKE_JOB_POOL_LINK CACHE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Add a -g flag for debug info in all CMAKE build types (debug, release, etc.)
add_compile_options(-g)

# Keep the frame pointer (rbp on x86-64) in every translation unit so that
# perf/eBPF/profilers can walk stacks via `--call-graph fp` instead of paying
# the cost of DWARF unwinding. Runtime cost on this workload is <1% (mutex /
# IPC / HTTP-framing dominated, not register-pressure-bound numeric loops);
# capture-time win is ~10x smaller perf.data and stacks that don't truncate
# across -O2 optimization boundaries. Fedora 38+ and Ubuntu 24.04+ ship
# distro-wide with this flag for the same reason.
add_compile_options(-fno-omit-frame-pointer)


# Allow config files (e.g. system jsoncpp) that use older cmake_minimum_required (CMake 4.x compatibility)
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Min policy version for included configs")
endif()

# Optional clang-tidy during build (same approach as tt-metal: lint = build with clang-tidy)
# Set only after FetchContent so _deps (tokenizers_cpp, sentencepiece, protobuf, etc.) are excluded.
option(CLANG_TIDY "Run clang-tidy during build" OFF)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

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

# Google Benchmark for microbenchmarks
FetchContent_Declare(googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
    GIT_SHALLOW    TRUE
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# GCC 12's libstdc++ still uses the C++17-deprecated std::get_temporary_buffer
# inside std::stable_sort, which gtest.cc triggers when sorting TestInfo*.
# Silence on the googletest targets only — keep -Wdeprecated-declarations active
# for our own code.
foreach(_gt_target gtest gtest_main gmock gmock_main)
    if(TARGET ${_gt_target})
        target_compile_options(${_gt_target} PRIVATE -Wno-deprecated-declarations)
    endif()
endforeach()

# xxHash for stable, cross-platform hashing (conversation prefix caching)
FetchContent_Declare(
    xxhash
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG        v0.8.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(xxhash)
add_library(xxhash_lib INTERFACE)
target_include_directories(xxhash_lib INTERFACE ${xxhash_SOURCE_DIR})

# BLAKE3 (official C library). The Dynamo frontend blake3-validates every file
# it fetches from a Model Descriptor Card (config.json, tokenizer, chat
# template, tokenizer_config); a placeholder/zero checksum makes it reject the
# model and the backend never becomes routable. We hash those files at
# registration time in src/dynamo/discovery.cpp. SOURCE_SUBDIR points at the
# repo's c/ project, which enables the C/ASM languages it needs in its own scope.
FetchContent_Declare(
    blake3
    GIT_REPOSITORY https://github.com/BLAKE3-team/BLAKE3.git
    GIT_TAG        1.5.4
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  c
)
FetchContent_MakeAvailable(blake3)

# xgrammar for structured output / guided decoding (JSON schema constrained generation)
FetchContent_Declare(xgrammar
    GIT_REPOSITORY https://github.com/mlc-ai/xgrammar.git
    GIT_TAG        v0.1.33
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(xgrammar)
if(NOT xgrammar_POPULATED)
    FetchContent_Populate(xgrammar)
    # xgrammar ships cmake/config.cmake that forces PYTHON_BINDINGS=ON; override it
    # in the source tree so its CMakeLists.txt picks up our settings.
    file(WRITE "${xgrammar_SOURCE_DIR}/cmake/config.cmake"
        "set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)\n"
        "set(XGRAMMAR_BUILD_CXX_TESTS OFF)\n"
        "set(XGRAMMAR_ENABLE_CPPTRACE OFF)\n"
        "set(XGRAMMAR_ENABLE_COVERAGE OFF)\n"
        "set(XGRAMMAR_ENABLE_INTERNAL_CHECK OFF)\n"
    )
    add_subdirectory(${xgrammar_SOURCE_DIR} ${xgrammar_BINARY_DIR})
    target_compile_options(xgrammar PRIVATE -Wno-deprecated-declarations -fno-lto)
endif()

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

# Ensure static libraries are built with -fPIC (required when linking into shared libs like tt-llm-engine)
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
find_package(Boost 1.71 REQUIRED)

# ZeroMQ for optional inter-server transport (alternative to raw TCP sockets)
# First try to find system-installed libzmq; if not found, build from source.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ IMPORTED_TARGET libzmq)
endif()
if(NOT ZMQ_FOUND)
    find_package(ZeroMQ QUIET)
    if(ZeroMQ_FOUND)
        set(ZMQ_FOUND TRUE)
    endif()
endif()
if(NOT ZMQ_FOUND)
    message(STATUS "libzmq not found on system; building from source via FetchContent")
    FetchContent_Declare(
        libzmq
        GIT_REPOSITORY https://github.com/zeromq/libzmq.git
        GIT_TAG        v4.3.5
        GIT_SHALLOW    TRUE
    )
    set(ZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(WITH_DOCS OFF CACHE BOOL "" FORCE)
    set(WITH_PERF_TOOL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(libzmq)
    set(ZMQ_FROM_SOURCE TRUE)
endif()
# cppzmq (header-only C++ bindings for libzmq)
FetchContent_Declare(
    cppzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG        v4.11.0
    GIT_SHALLOW    TRUE
)
set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cppzmq)

# Option to enable tt-llm-engine (formerly tt-blaze pipeline_manager)
option(ENABLE_BLAZE "Enable tt-llm-engine support" OFF)
# Real shmem migration (MigrationLayerClientAdapter + migration_client static lib).
# mock_pipeline CI builds use --blaze only; disaggregated pipeline_manager runs
# need --blaze-with-migration (implies ENABLE_BLAZE).
option(ENABLE_BLAZE_MIGRATION "Build real shmem migration client (requires ENABLE_BLAZE)" OFF)
if(ENABLE_BLAZE_MIGRATION AND NOT ENABLE_BLAZE)
    message(FATAL_ERROR "ENABLE_BLAZE_MIGRATION=ON requires ENABLE_BLAZE=ON (use build.sh --blaze-with-migration)")
endif()

if(ENABLE_BLAZE)
    # Prefer a pre-built tt-llm-engine discovered via find_package — this is
    # the path used by the Docker build, which pulls /opt/tt-llm-engine out
    # of the tt-llm-engine builder image and sets CMAKE_PREFIX_PATH so the
    # package is locatable. Fall back to building the in-tree submodule for
    # local-dev workflows that don't go through Docker.
    find_package(tt-llm-engine CONFIG QUIET)
    if(tt-llm-engine_FOUND)
        message(STATUS "tt-llm-engine: using pre-built package at ${tt-llm-engine_DIR}")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tt-llm-engine/CMakeLists.txt")
        # Legacy path — local dev with the submodule checked out under cpp_server/.
        # The narrowing-warning flag is only needed when we actually compile
        # tt-llm-engine from source; the find_package path consumes a prebuilt
        # .so so the warning is irrelevant there.
        set(DS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        # tt-llm-engine has C++11 braced-init narrowing that GCC only warns on, but
        # the tt-metal clang-20 toolchain rejects as a hard error. Downgrade it to a
        # warning for the sub-build only; restore our stricter flags afterwards.
        set(_tt_saved_cxx_flags "${CMAKE_CXX_FLAGS}")
        string(APPEND CMAKE_CXX_FLAGS " -Wno-c++11-narrowing")
        add_subdirectory(tt-llm-engine)
        set(CMAKE_CXX_FLAGS "${_tt_saved_cxx_flags}")
        message(STATUS "tt-llm-engine: building from in-tree submodule (legacy path)")
    else()
        message(FATAL_ERROR
            "ENABLE_BLAZE=ON but tt-llm-engine is neither installed "
            "(find_package failed) nor present in source tree at "
            "${CMAKE_CURRENT_SOURCE_DIR}/tt-llm-engine. "
            "Either ensure the prebuilt image's /opt/tt-llm-engine is on "
            "CMAKE_PREFIX_PATH, or initialize the tt-llm-engine submodule.")
    endif()
else()
    message(STATUS "tt-llm-engine: DISABLED (use -DENABLE_BLAZE=ON to enable)")
endif()

# Mooncake Transfer Engine (TCP + RDMA) is vendored as a top-level git submodule
# at third_party/Mooncake (pinned to release v0.3.11.post1) and built directly
# here via its standalone mooncake-transfer-engine project. It is independent of
# tt-llm-engine. RDMA (libibverbs) is always compiled in -- the standalone build
# has no library-level toggle to disable it.
#
# System dependencies are assumed to already be installed (provision them once
# with third_party/Mooncake/dependencies.sh): libibverbs-dev, libnuma-dev,
# libgoogle-glog-dev, libgflags-dev, libjsoncpp-dev, libyaml-cpp-dev, and an
# installed yalantinglibs (resolved via find_package(yalantinglibs CONFIG)).
option(ENABLE_MOONCAKE "Build the Mooncake Transfer Engine transport (third_party/Mooncake)" OFF)
if(ENABLE_MOONCAKE)
    set(_mooncake_root "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Mooncake")
    if(NOT EXISTS "${_mooncake_root}/mooncake-transfer-engine/CMakeLists.txt")
        message(FATAL_ERROR
            "ENABLE_MOONCAKE=ON but third_party/Mooncake is not checked out. Run: "
            "git submodule update --init --recursive "
            "tt-media-server/cpp_server/third_party/Mooncake")
    endif()
    # Build only the Transfer Engine -- not the store, Python/Rust bindings,
    # examples, benchmarks or Mooncake's own unit tests. Use the HTTP metadata
    # server (no etcd/redis).
    set(USE_TCP           ON  CACHE BOOL "" FORCE)
    set(USE_HTTP          ON  CACHE BOOL "" FORCE)
    set(USE_ETCD          OFF CACHE BOOL "" FORCE)
    set(USE_REDIS         OFF CACHE BOOL "" FORCE)
    set(WITH_RUST_EXAMPLE OFF CACHE BOOL "" FORCE)
    set(BUILD_UNIT_TESTS  OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARK   OFF CACHE BOOL "" FORCE)
    # Mooncake's source has C++11 braced-init narrowing (e.g. int -> uint16_t in
    # tcp_transport.cpp) that GCC (its usual compiler) only warns on, but the
    # tt-metal clang-20 toolchain rejects as a hard error. Downgrade it to a
    # warning for the sub-build only; restore our stricter flags afterwards.
    set(_tt_saved_cxx_flags "${CMAKE_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " -Wno-c++11-narrowing")
    add_subdirectory("${_mooncake_root}/mooncake-transfer-engine"
                     "${CMAKE_CURRENT_BINARY_DIR}/mooncake-transfer-engine")
    set(CMAKE_CXX_FLAGS "${_tt_saved_cxx_flags}")
    message(STATUS "Mooncake transport: ENABLED (third_party/Mooncake, RDMA always on)")
else()
    message(STATUS "Mooncake transport: DISABLED (use -DENABLE_MOONCAKE=ON to enable)")
endif()

# Option for Tracy support
option(ENABLE_TRACY "Enable Tracy profiling instrumentation" OFF)

if(ENABLE_TRACY AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR
        "Tracy profiling (ENABLE_TRACY=ON) must not be used in Release builds: "
        "the Tracy client spawns background threads that degrade throughput on "
        "machines with limited cores. Use a Debug build instead: "
        "./build.sh --tracy")
endif()

# Option for Kafka support
option(KAFKA_ENABLED "Enable Kafka messaging support" OFF)

if(ENABLE_TRACY)
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/tenstorrent/tracy.git
        GIT_TAG v0.10-tt.0
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

# stb_image_write: in-memory PNG/JPEG encoder used by image_codec.
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        31c1ad37456438565541f4919958214b6e762fb4
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
    FetchContent_Populate(stb)
endif()
add_library(stb_image_write INTERFACE)
target_include_directories(stb_image_write SYSTEM INTERFACE ${stb_SOURCE_DIR})

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

# LLM runner static library (C++ namespace tt::runners::llm_engine)
set(LLM_RUNNER_LIB_SOURCES
    src/domain/llm/sequence.cpp
    src/domain/llm/sampling_params.cpp
    src/utils/logger.cpp
    src/utils/conversation_hasher.cpp
    src/config/settings.cpp
    src/utils/tokenizers/tokenizer.cpp
    src/utils/tokenizers/deepseek_tokenizer.cpp
    src/utils/tokenizers/llama_tokenizer.cpp
    src/utils/tokenizers/tokenizer_config_loader.cpp
    src/services/memory_services/memory_manager.cpp
    src/runtime/runners/guided_decoder_manager.cpp
    src/services/service_registry.cpp
    src/runtime/runners/runner_registry.cpp
    src/api/route_registry.cpp
)

# Add tt-llm-engine-dependent sources only when ENABLE_BLAZE is ON
if(ENABLE_BLAZE)
    list(APPEND LLM_RUNNER_LIB_SOURCES
        src/runtime/runners/blaze_runner/blaze_decode_runner.cpp
        src/runtime/runners/blaze_runner/blaze_prefill_runner.cpp
        src/runtime/runners/blaze_runner/blaze_scheduler_factory.cpp
    )
    message(STATUS "llm_runner_lib: including tt-llm-engine-dependent sources (ENABLE_BLAZE=ON)")
endif()

add_library(llm_runner_lib ${LLM_RUNNER_LIB_SOURCES})
target_include_directories(llm_runner_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TT_METAL_INCLUDE_DIRS}
)
# Link tokenizers_cpp, JsonCpp, xgrammar (guided decoding), and xxhash (stable hashing)
target_link_libraries(llm_runner_lib PUBLIC tokenizers_cpp JsonCpp::JsonCpp xgrammar xxhash_lib)

# Link tt-llm-engine when ENABLE_BLAZE is ON: prefer Full (SocketPipeline + TT::Metalium),
# fall back to Core (mock pipeline only).
if(ENABLE_BLAZE)
    if(TARGET TtLlmEngine::Full)
        target_link_libraries(llm_runner_lib PUBLIC TtLlmEngine::Full)
        message(STATUS "Linking llm_runner_lib against TtLlmEngine::Full")
    elseif(TARGET TtLlmEngine::Core)
        target_link_libraries(llm_runner_lib PUBLIC TtLlmEngine::Core)
        message(STATUS "Linking llm_runner_lib against TtLlmEngine::Core")
    endif()
    target_compile_definitions(llm_runner_lib PUBLIC ENABLE_BLAZE)
    # Detect where the tt-llm-engine source tree lives. Two layouts are supported:
    #   1. Monolithic flow (main's Dockerfile.blaze): cpp_server/tt-llm-engine/ —
    #      cloned in-tree during the Docker build.
    #   2. Slim flow (v2 Dockerfile.blaze): /opt/tt-llm-engine-src/ — COPY'd from
    #      the prebuilt tt-llm-engine image's source tree.
    # Both layouts expose the same subdirectories (src/, disaggregation/, tt-metal/),
    # so a single root variable lets every consumer-of-engine-source path adapt.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tt-llm-engine/CMakeLists.txt")
        set(_TT_LLM_ENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tt-llm-engine")
        message(STATUS "tt-llm-engine source: in-tree at ${_TT_LLM_ENGINE_ROOT}")
    elseif(EXISTS "/opt/tt-llm-engine-src/CMakeLists.txt")
        set(_TT_LLM_ENGINE_ROOT "/opt/tt-llm-engine-src")
        message(STATUS "tt-llm-engine source: slim engine image at ${_TT_LLM_ENGINE_ROOT}")
    else()
        message(FATAL_ERROR
            "ENABLE_BLAZE=ON but tt-llm-engine source not found. Expected either "
            "${CMAKE_CURRENT_SOURCE_DIR}/tt-llm-engine/ (in-tree clone, monolithic flow) "
            "or /opt/tt-llm-engine-src/ (slim engine image flow).")
    endif()

    # Header-only MigrationClientInterface implementations (MigrationLayerClientAdapter,
    # MockMigrationClient) live under the engine's src tree rather than its public
    # include tree; blaze_utils.hpp instantiates them, so expose that path.
    target_include_directories(llm_runner_lib PUBLIC
        ${_TT_LLM_ENGINE_ROOT}/src)

    if(ENABLE_BLAZE_MIGRATION)
        target_compile_definitions(llm_runner_lib PUBLIC ENABLE_BLAZE_MIGRATION)
        # MigrationLayerClientAdapter forwards to migration_worker::MigrationLayerClient,
        # whose implementation lives in the disaggregation/migration project. That
        # project is a standalone CMake project() (own CPM cache + find_package(MPI)),
        # so it can't be add_subdirectory'd here. Its client is lightweight though
        # (just POSIX shmem queues), so we compile the two needed translation units
        # directly into a small static lib and link it in. There is a single concrete
        # implementation (no mock/real .so swap), so static linking is sufficient for
        # both Core and LD_PRELOAD'd Full runs.
        set(_MIG_DIR ${_TT_LLM_ENGINE_ROOT}/disaggregation/migration)
        set(_MIG_TT_STL "${_TT_LLM_ENGINE_ROOT}/tt-metal/tt_stl")
        add_library(migration_client STATIC
            ${_MIG_DIR}/src/migration_client.cpp
            ${_MIG_DIR}/src/file_lock_guard.cpp)
        set_target_properties(migration_client PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_include_directories(migration_client PUBLIC
            ${_MIG_DIR}/include
            ${_MIG_DIR}/src/worker
            ${_TT_LLM_ENGINE_ROOT}/src/common
            ${TT_METAL_INCLUDE_DIRS}
            ${_MIG_TT_STL})
        target_link_libraries(migration_client PUBLIC spdlog::spdlog PRIVATE rt)
        target_link_libraries(llm_runner_lib PUBLIC migration_client)
        message(STATUS "Blaze migration client: ENABLED (ENABLE_BLAZE_MIGRATION=ON)")
    else()
        message(STATUS "Blaze migration client: DISABLED (use build.sh --blaze-with-migration for pipeline_manager shmem migration)")
    endif()
else()
    message(STATUS "Skipping tt-llm-engine linking (ENABLE_BLAZE=OFF)")
endif()
if(USE_EXTERNAL_FMT)
    target_link_libraries(llm_runner_lib PUBLIC Boost::boost spdlog::spdlog fmt::fmt)
else()
    target_link_libraries(llm_runner_lib PUBLIC Boost::boost spdlog::spdlog)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(llm_runner_lib PUBLIC rt)
endif()
target_compile_features(llm_runner_lib PUBLIC cxx_std_20)
target_compile_options(llm_runner_lib PRIVATE -Wall -Wextra -Wpedantic)

if(ENABLE_TRACY)
    target_link_libraries(llm_runner_lib PUBLIC tracy_client)
endif()

option(SANITIZE_THREAD "Build with ThreadSanitizer (TSan) for data-race detection" OFF)
option(SANITIZE_ADDRESS "Build with AddressSanitizer (ASan) and LeakSanitizer for memory errors and leak detection" OFF)
if(SANITIZE_THREAD AND SANITIZE_ADDRESS)
    message(FATAL_ERROR "SANITIZE_THREAD and SANITIZE_ADDRESS are mutually exclusive; enable only one.")
endif()
if(SANITIZE_THREAD)
    target_compile_options(llm_runner_lib PUBLIC -fsanitize=thread -g)
    target_link_options(llm_runner_lib PUBLIC -fsanitize=thread)
endif()
if(SANITIZE_ADDRESS)
    set(ASAN_FLAGS -fsanitize=address -fno-omit-frame-pointer -g)
    target_compile_options(llm_runner_lib PUBLIC ${ASAN_FLAGS})
    target_link_options(llm_runner_lib PUBLIC -fsanitize=address)
endif()

# Transport layer placeholder (#3890, "Custom backend for DRAM access via UMD").
# Models the Transfer Engine's two mechanisms at the cpp_server level:
#   - storage  : IStorageBackend + host/device-DRAM backends, the device one
#                being the custom UMD backend (umd_device_access wrapper);
#   - transport: the Mooncake Transfer Engine wrapper (TCP/RDMA).
# Plus the migration-worker spike (write-on-sender / transfer / verify-on-
# receiver). Self-contained stubs with no Mooncake/tt-metal deps, so this builds
# in every configuration; the real backends land behind the pimpls.
add_library(transport_lib
    src/transport/host_dram_storage_backend.cpp
    src/transport/device_dram_storage_backend.cpp
    src/transport/umd_device_access.cpp
    src/transport/mooncake_transfer_engine.cpp
    src/transport/mooncake_migration_worker.cpp
    src/transport/peer_discovery_service.cpp
    # Real KV-cache addressing: physical mirror layout + sender/receiver views +
    # multi-device UMD I/O. Dependency-free host code (no tt-metal/Mooncake), so
    # transport_lib still builds in every configuration.
    src/transport/kv_cache_layout.cpp
    src/transport/kv_cache_mirror.cpp
    src/transport/remote_region.cpp
    src/transport/multi_device_umd.cpp
    src/transport/kv_table_adapter.cpp
    src/transport/kv_chunk_address_table_adapter.cpp
    src/transport/kv_control_message.cpp
    src/transport/kv_control_channel.cpp
    src/transport/mooncake_kv_sender.cpp
    src/transport/mooncake_kv_receiver.cpp
    src/transport/kv_migration_orchestrator.cpp
    src/transport/kv_migration_multi_host_sender.cpp
    src/utils/logger.cpp
)
target_include_directories(transport_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(transport_lib PUBLIC spdlog::spdlog)
if(USE_EXTERNAL_FMT)
    target_link_libraries(transport_lib PUBLIC fmt::fmt)
endif()
# Real UMD device-DRAM backend (umd_device_access.cpp). Uses the coexistence
# UmdDevice from tt-llm-engine/disaggregation (raw UMD Cluster, NO start_device
# / NO CHIP_IN_USE flock / NO dispatch firmware), so the migration worker shares
# device DRAM with a live inference engine. We compile the two sources it needs
# directly (device_io.cpp + tt-metal's metal_soc_descriptor.cpp) and link UMD —
# NOT libtt_metal (which would take exclusive chip ownership and JIT-build
# dispatch kernels; see the device-mode failures this replaces). Requires a
# BUILT tt-metal (UMD .so + yaml-cpp); a headers-only / unbuilt tree leaves
# USE_METAL_CPP_LIB undefined so umd_device_access.cpp uses its no-op fallback
# and transport_lib still builds in every configuration.
if(TT_METAL_API_FOUND AND TT_METAL_INCLUDE_DIRS)
    set(_TTM_BUILD "")
    if(EXISTS "${TT_METAL_HOME}/build_Release/lib")
        set(_TTM_BUILD "${TT_METAL_HOME}/build_Release")
    elseif(EXISTS "${TT_METAL_HOME}/build/lib")
        set(_TTM_BUILD "${TT_METAL_HOME}/build")
    endif()

    # UMD + support libs the coexistence device path resolves against (no libtt_metal).
    find_library(_KV_UMD_LIB   tt-umd   PATHS "${_TTM_BUILD}/lib"                 NO_DEFAULT_PATH)
    find_library(_KV_TTSTL_LIB tt_stl   PATHS "${_TTM_BUILD}/tt_stl"              NO_DEFAULT_PATH)
    find_library(_KV_TRACY_LIB tracy    PATHS "${_TTM_BUILD}/lib"                 NO_DEFAULT_PATH)
    find_library(_KV_YAML_LIB  yaml-cpp PATHS "${_TTM_BUILD}/_deps/yaml-cpp-build" NO_DEFAULT_PATH)

    if(_TTM_BUILD AND _KV_UMD_LIB AND _KV_TTSTL_LIB AND _KV_TRACY_LIB AND _KV_YAML_LIB AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tt-llm-engine/disaggregation/migration/src/worker/device_io.cpp")
        target_compile_definitions(transport_lib PUBLIC USE_METAL_CPP_LIB)
        target_include_directories(transport_lib PUBLIC ${TT_METAL_INCLUDE_DIRS})

        # device_io.cpp (UmdDevice) + metal_soc_descriptor.cpp compiled directly.
        set(_KV_MIG "${CMAKE_CURRENT_SOURCE_DIR}/tt-llm-engine/disaggregation/migration")
        target_sources(transport_lib PRIVATE
            ${_KV_MIG}/src/worker/device_io.cpp
            ${TT_METAL_HOME}/tt_metal/llrt/metal_soc_descriptor.cpp)
        # Third-party TUs: silence tt-metal/UMD header warnings (we're not -Werror,
        # but keep the log clean) without relaxing diagnostics on our own code.
        set_source_files_properties(
            ${_KV_MIG}/src/worker/device_io.cpp
            ${TT_METAL_HOME}/tt_metal/llrt/metal_soc_descriptor.cpp
            PROPERTIES COMPILE_OPTIONS "-w")

        # device_io.hpp + the tt-metal-internal headers metal_soc_descriptor.cpp
        # needs (mirrors the migration device target's include set). cpmcache
        # globs match how tt-metal vendored these deps.
        file(GLOB _KV_NLOHMANN "${TT_METAL_HOME}/.cpmcache/nlohmann_json/*/include")
        file(GLOB _KV_REFLECT  "${TT_METAL_HOME}/.cpmcache/reflect/f*")
        file(GLOB _KV_TASKFLOW "${TT_METAL_HOME}/.cpmcache/taskflow/5*")
        file(GLOB _KV_YAMLINC  "${TT_METAL_HOME}/.cpmcache/yaml-cpp/*/include")
        target_include_directories(transport_lib SYSTEM PRIVATE
            "${_KV_MIG}/include"
            "${TT_METAL_HOME}/tt_metal/impl"
            "${TT_METAL_HOME}/tt_metal/impl/debug"
            "${TT_METAL_HOME}/tt_metal/impl/profiler"
            "${TT_METAL_HOME}/tt_metal/llrt"
            "${TT_METAL_HOME}/tt_metal/common"
            "${TT_METAL_HOME}/tt_metal/third_party/umd/src/firmware/riscv"
            "${_TTM_BUILD}/tt_metal/impl"
            ${_KV_NLOHMANN} ${_KV_REFLECT} ${_KV_TASKFLOW} ${_KV_YAMLINC})

        # Full-path .so links propagate rpath to the e2e executables automatically.
        target_link_libraries(transport_lib PUBLIC
            ${_KV_UMD_LIB} ${_KV_TTSTL_LIB} ${_KV_TRACY_LIB} ${_KV_YAML_LIB})
        if(USE_EXTERNAL_FMT)
            target_link_libraries(transport_lib PUBLIC fmt::fmt)
        endif()
        message(STATUS "transport_lib: coexistence UMD device IO enabled (UmdDevice; libtt_metal NOT linked)")
    else()
        message(STATUS "transport_lib: tt-metal headers found but built UMD libs missing — device IO falls back to no-op (set TT_METAL_HOME to a BUILT tree)")
    endif()
endif()
# Real Mooncake transport (mooncake_transfer_engine.cpp) is compiled against the
# Mooncake Transfer Engine when it is part of the build (ENABLE_MOONCAKE builds
# the `transfer_engine` target from third_party/Mooncake); otherwise it falls
# back to a no-op (see the TT_TRANSPORT_WITH_MOONCAKE guard in that file).
if(TARGET transfer_engine)
    target_compile_definitions(transport_lib PUBLIC TT_TRANSPORT_WITH_MOONCAKE)
    target_include_directories(transport_lib PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Mooncake/mooncake-transfer-engine/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Mooncake/mooncake-common/include")
    target_link_libraries(transport_lib PUBLIC transfer_engine)
    message(STATUS "transport_lib: building with Mooncake transport")
endif()
# Real KvChunkAddressTable adapter (kv_chunk_address_table_adapter.cpp). OFF by
# default — the file then compiles its no-op fallback and callers use the
# InMemoryKvTable. When ON, it wraps the real disaggregation table loaded from
# protobuf.
#
# Rather than build tt-llm-engine's standalone `migration` CMake project (its own
# project()/CPM/find_package(MPI), only buildable via an ExternalProject inside a
# test-gated block), we mirror the lightweight migration_client pattern used
# above: compile the exact translation units the adapter needs directly into a
# small cpp_server-owned static lib. The table impl is the metal-free synthetic
# stub (same public API as tt-metal's, only read_device_chunk() throws — the
# adapter never calls it), so this needs NO libtt_metal, NO MPI, NO MPI test
# harness — only tt-metal headers + protobuf. Requires TT_METAL_HOME (tt-metal
# headers); see TT_METAL_API_FOUND above.
option(ENABLE_KV_TABLE "Build the real KvChunkAddressTable adapter (tt-metal + protobuf)" OFF)
if(ENABLE_KV_TABLE)
    if(NOT (TT_METAL_API_FOUND AND TT_METAL_INCLUDE_DIRS))
        message(FATAL_ERROR
            "ENABLE_KV_TABLE=ON requires tt-metal headers, but TT_METAL_API_FOUND "
            "is OFF. Set TT_METAL_HOME to a tt-metal checkout/build (e.g. "
            "tt-llm-engine/tt-metal) so tt_metal/api is on the include path.")
    endif()
    # tt-metal's spdlog headers (pulled in transitively by the table headers)
    # require external fmt; a BUILT tt-metal provides both that fmt and the
    # protobuf runtime this path links. With external fmt, spdlog::spdlog carries
    # SPDLOG_FMT_EXTERNAL PUBLIC, so the adapter TU in transport_lib compiles
    # against the same fmt without any extra per-target define.
    if(NOT USE_EXTERNAL_FMT)
        message(FATAL_ERROR
            "ENABLE_KV_TABLE=ON requires a BUILT tt-metal that provides fmt "
            "(USE_EXTERNAL_FMT). Build tt-metal so "
            "${TT_METAL_HOME}/build_Release/lib/cmake/fmt exists.")
    endif()

    # Protobuf: prefer tt-metal's vendored build (matches the .proto's runtime),
    # fall back to the system package. protobuf_generate_cpp uses the matching
    # protoc, so generation + link stay self-consistent either way.
    find_package(Protobuf QUIET
        HINTS "${TT_METAL_HOME}/build_Release/_deps/protobuf-build"
              "${TT_METAL_HOME}/build_Release/_deps/protobuf-build/lib/cmake/protobuf"
              "${TT_METAL_HOME}/build/_deps/protobuf-build")
    if(NOT Protobuf_FOUND)
        find_package(Protobuf REQUIRED)
        message(STATUS "ENABLE_KV_TABLE: protobuf from system package (${Protobuf_VERSION})")
    else()
        message(STATUS "ENABLE_KV_TABLE: protobuf from tt-metal build (${Protobuf_VERSION})")
    endif()

    set(_KV_MIG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tt-llm-engine/disaggregation/migration")
    set(_KV_TTM "${CMAKE_CURRENT_SOURCE_DIR}/tt-llm-engine/tt-metal")
    set(_KV_PROTO_FILE
        "${_KV_TTM}/tt_metal/impl/experimental/disaggregation/protobuf/kv_chunk_address_table.proto")
    protobuf_generate_cpp(_KV_PROTO_SRCS _KV_PROTO_HDRS "${_KV_PROTO_FILE}")

    # tt-metal sources #include "protobuf/kv_chunk_address_table.pb.h"; stage the
    # generated header under a protobuf/ subdir to match that include path.
    set(_KV_PROTO_INC "${CMAKE_CURRENT_BINARY_DIR}/kv_table_proto_include")
    file(MAKE_DIRECTORY "${_KV_PROTO_INC}/protobuf")
    add_custom_command(
        OUTPUT "${_KV_PROTO_INC}/protobuf/kv_chunk_address_table.pb.h"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_BINARY_DIR}/kv_chunk_address_table.pb.h"
            "${_KV_PROTO_INC}/protobuf/kv_chunk_address_table.pb.h"
        DEPENDS ${_KV_PROTO_HDRS}
        COMMENT "Staging kv_chunk_address_table.pb.h under protobuf/ include alias")

    add_library(kv_table_lib STATIC
        ${_KV_MIG_DIR}/src/kv_chunk_address_table_synthetic.cpp
        ${_KV_MIG_DIR}/src/fabric_types_shim.cpp
        ${_KV_TTM}/tt_metal/impl/experimental/disaggregation/kv_chunk_address_table_protobuf.cpp
        ${_KV_PROTO_SRCS}
        "${_KV_PROTO_INC}/protobuf/kv_chunk_address_table.pb.h"
    )
    set_target_properties(kv_table_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # Every source here is generated (protobuf) or tt-metal third-party — none of it
    # is cpp_server code we lint. Clear the per-target CXX_CLANG_TIDY so --clang-tidy
    # builds don't flag protobuf's snake_case identifiers as naming violations. This
    # mirrors the -w below, which only silences the compiler (not clang-tidy).
    set_target_properties(kv_table_lib PROPERTIES CXX_CLANG_TIDY "")
    target_include_directories(kv_table_lib PUBLIC
        ${TT_METAL_INCLUDE_DIRS}
        "${_KV_TTM}/tt_metal/impl"          # experimental/disaggregation/...protobuf.hpp
        ${CMAKE_CURRENT_BINARY_DIR}          # generated kv_chunk_address_table.pb.h
        ${_KV_PROTO_INC}                     # protobuf/ alias
    )
    # tt-metal's spdlog headers expect external fmt; PRIVATE so the define stays
    # on kv_table_lib's own sources and does not leak onto transport_lib's other
    # TUs (which use cpp_server's spdlog). fmt::fmt provides the symbols.
    target_compile_definitions(kv_table_lib PRIVATE SPDLOG_FMT_EXTERNAL)
    target_link_libraries(kv_table_lib PUBLIC protobuf::libprotobuf fmt::fmt)
    # tt-metal headers trip -Wall/-Wpedantic (deprecated tt::stl, etc.); these are
    # third-party sources, so silence rather than gate cpp_server's warnings on them.
    target_compile_options(kv_table_lib PRIVATE -w)

    target_link_libraries(transport_lib PUBLIC kv_table_lib)
    target_compile_definitions(transport_lib PUBLIC TT_TRANSPORT_WITH_KV_TABLE)
    message(STATUS "transport_lib: building with real KvChunkAddressTable (ENABLE_KV_TABLE=ON)")
endif()

target_compile_features(transport_lib PUBLIC cxx_std_20)
target_compile_options(transport_lib PRIVATE -Wall -Wextra -Wpedantic)

# Two-process acceptance harness for the #3890 migration path (the "Two
# galaxies" validation in mooncake/poc-transfer-engine/adr-mooncake-backend.md). It is a manual
# sender/receiver driver, not a ctest case, and is only built when Mooncake is
# in the build (the engine is otherwise a no-op). Run it via
# tests/e2e/scripts/run_transport_migration_e2e.sh.
if(TARGET transfer_engine)
    add_executable(transport_migration_e2e
        tests/e2e/transport_migration_e2e.cpp)
    target_link_libraries(transport_migration_e2e PRIVATE transport_lib)
    target_include_directories(transport_migration_e2e PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    message(STATUS "transport_migration_e2e: ENABLED (Mooncake acceptance harness)")

    # #4209 migration-worker discovery PoC (see src/transport/README.md).
    add_executable(migration_worker_discovery
        tests/e2e/migration_worker_discovery.cpp)
    target_link_libraries(migration_worker_discovery PRIVATE transport_lib)
    target_include_directories(migration_worker_discovery PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    message(STATUS "migration_worker_discovery: ENABLED (#4209 discovery PoC)")

    # #4294 production worker entry point is defined later, inside the
    # KAFKA_ENABLED block, because it links kv_migration_worker for the Kafka
    # request loop. See `if(KAFKA_ENABLED)` further down.

    # Real KV-cache migration (table-addressed mirror + orchestration over a
    # TCP control channel). Self-contained TCP control transport, links only
    # transport_lib. host mode runs with no hardware; device mode needs --blaze;
    # real-table (--table <pb>) needs -DENABLE_KV_TABLE=ON.
    add_executable(transport_kv_migration_e2e
        tests/e2e/transport_kv_migration_e2e.cpp)
    # GTest::gtest (not gtest_main): the binary has a custom main() that dispatches
    # --role sender|receiver workers before falling through to RUN_ALL_TESTS.
    target_link_libraries(transport_kv_migration_e2e PRIVATE transport_lib GTest::gtest)
    target_include_directories(transport_kv_migration_e2e PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    message(STATUS "transport_kv_migration_e2e: ENABLED (real KV migration harness)")
endif()

# Unit tests (link llm_runner_lib)
enable_testing()

# Transport layer placeholder smoke test (also forces transport_lib to link).
add_executable(transport_test tests/unit/transport/transport_test.cpp)
target_link_libraries(transport_test PRIVATE transport_lib GTest::gtest_main)
target_include_directories(transport_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# KV-cache addressing unit tests: mirror layout / offset math, sender-receiver
# offset parity, multi-device UMD dispatch. Host-only, runs in any build config.
add_executable(kv_cache_addressing_test
    tests/unit/transport/kv_cache_addressing_test.cpp)
target_link_libraries(kv_cache_addressing_test PRIVATE transport_lib GTest::gtest_main)
target_include_directories(kv_cache_addressing_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Table -> KvChunkLocation adapter (host/slot/position filtering + device-group
# fan-out). Host-only, runs in any build config.
add_executable(kv_table_adapter_test
    tests/unit/transport/kv_table_adapter_test.cpp)
target_link_libraries(kv_table_adapter_test PRIVATE transport_lib GTest::gtest_main)
target_include_directories(kv_table_adapter_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Control-plane message codec + channel (table exchange, request, mirror name,
# done-marker, ack). Host-only loopback, runs in any build config.
add_executable(kv_control_channel_test
    tests/unit/transport/kv_control_channel_test.cpp)
target_link_libraries(kv_control_channel_test PRIVATE transport_lib GTest::gtest_main)
target_include_directories(kv_control_channel_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# End-to-end KV migration data plane (sender -> simulated one-sided wire ->
# receiver -> device) with in-memory fakes. Host-only, runs in any build config.
add_executable(mooncake_kv_migration_test
    tests/unit/transport/mooncake_kv_migration_test.cpp)
target_link_libraries(mooncake_kv_migration_test PRIVATE transport_lib GTest::gtest_main)
target_include_directories(mooncake_kv_migration_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Control-channel orchestration: drives BeginMigration/MirrorReady/DoneMarker/Ack
# end to end over a blocking loopback channel (receiver on its own thread).
add_executable(kv_migration_orchestrator_test
    tests/unit/transport/kv_migration_orchestrator_test.cpp)
target_link_libraries(kv_migration_orchestrator_test PRIVATE transport_lib GTest::gtest_main)
target_include_directories(kv_migration_orchestrator_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Real-table adapter test: loads an actual serialized KvChunkAddressTable (.pb)
# and checks real NoC addresses resolve. Only built when ENABLE_KV_TABLE=ON
# (otherwise the adapter is the no-op fallback). Reads KV_TABLE_PB env var, else
# the default below; SKIPs if neither file is present.
if(ENABLE_KV_TABLE)
    add_executable(kv_chunk_address_table_adapter_test
        tests/unit/transport/kv_chunk_address_table_adapter_test.cpp)
    target_link_libraries(kv_chunk_address_table_adapter_test PRIVATE transport_lib GTest::gtest_main)
    target_include_directories(kv_chunk_address_table_adapter_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    # Repo-root tables emitted by the model runner (cpp_server -> tt-media-server -> repo root).
    target_compile_definitions(kv_chunk_address_table_adapter_test PRIVATE
        KV_TABLE_PB_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/../../decoder_kv_table.pb")
endif()

add_executable(dynamo_protocol_test
    tests/unit/dynamo/dynamo_protocol_test.cpp
    src/dynamo/dynamo_protocol.cpp
    src/utils/logger.cpp
)
target_link_libraries(dynamo_protocol_test PRIVATE
    Drogon::Drogon
    JsonCpp::JsonCpp
    spdlog::spdlog
    GTest::gtest_main)
target_include_directories(dynamo_protocol_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(ENABLE_BLAZE)
    add_executable(blaze_decode_runner_test
        tests/integration/runners/blaze_decode_runner_test.cpp
        src/runtime/worker/single_process_worker_metrics.cpp
        src/runtime/worker/worker_metrics_shm.cpp
    )
    target_link_libraries(blaze_decode_runner_test PRIVATE llm_runner_lib GTest::gtest_main)
    target_include_directories(blaze_decode_runner_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    add_executable(blaze_prefill_runner_test
        tests/integration/runners/blaze_prefill_runner_test.cpp
        src/runtime/worker/single_process_worker_metrics.cpp
        src/runtime/worker/worker_metrics_shm.cpp
    )
    target_link_libraries(blaze_prefill_runner_test PRIVATE llm_runner_lib GTest::gtest_main)
    target_include_directories(blaze_prefill_runner_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    add_executable(mock_schedulers_test
        tests/unit/runners/mock_schedulers_test.cpp
    )
    target_link_libraries(mock_schedulers_test PRIVATE llm_runner_lib GTest::gtest_main)
    target_include_directories(mock_schedulers_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

add_executable(sequence_test tests/unit/model/sequence_test.cpp)
target_link_libraries(sequence_test PRIVATE
    llm_runner_lib
    cereal::cereal
    GTest::gtest_main)
target_include_directories(sequence_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${cereal_SOURCE_DIR}/include
)

add_executable(lockfree_queue_test tests/unit/data_structures/lockfree_queue_test.cpp)
target_link_libraries(lockfree_queue_test PRIVATE GTest::gtest_main)
target_include_directories(lockfree_queue_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(tokenizer_test tests/unit/model/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE llm_runner_lib GTest::gtest_main)

add_executable(conversation_hasher_test tests/unit/model/conversation_hasher_test.cpp)
target_link_libraries(conversation_hasher_test PRIVATE llm_runner_lib GTest::gtest_main)

add_executable(cancel_queue_test tests/unit/data_structures/cancel_queue_test.cpp)
target_link_libraries(cancel_queue_test PRIVATE llm_runner_lib GTest::gtest_main)
target_include_directories(cancel_queue_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(structured_output_test tests/unit/api/structured_output_test.cpp)
target_link_libraries(structured_output_test PRIVATE llm_runner_lib GTest::gtest_main)
target_include_directories(structured_output_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(image_generate_request_test tests/unit/api/image_generate_request_test.cpp)
target_link_libraries(image_generate_request_test PRIVATE
    JsonCpp::JsonCpp
    GTest::gtest_main
)
target_include_directories(image_generate_request_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(media_payload_ipc_test tests/integration/sessions/media_payload_ipc_test.cpp)
target_link_libraries(media_payload_ipc_test PRIVATE llm_runner_lib GTest::gtest_main)
target_include_directories(media_payload_ipc_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(worker_metrics_shm_test
    tests/unit/workers/worker_metrics_shm_test.cpp
    src/runtime/worker/worker_metrics_shm.cpp
)
target_link_libraries(worker_metrics_shm_test PRIVATE llm_runner_lib GTest::gtest_main)
target_include_directories(worker_metrics_shm_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(model_service_registry_test
    tests/unit/services/model_service_registry_test.cpp
    src/services/service_container.cpp
)
target_link_libraries(model_service_registry_test PRIVATE llm_runner_lib GTest::gtest_main)

add_executable(health_controller_test
    tests/unit/api/health_controller_test.cpp
    src/api/health_controller.cpp
    src/services/service_container.cpp
)
target_link_libraries(health_controller_test PRIVATE
    llm_runner_lib
    Drogon::Drogon
    cereal::cereal
    JsonCpp::JsonCpp
    GTest::gtest_main)
target_include_directories(health_controller_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(concurrent_map_test tests/unit/data_structures/concurrent_map_test.cpp)
target_link_libraries(concurrent_map_test PRIVATE GTest::gtest_main)
target_include_directories(concurrent_map_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(session_test
    tests/unit/services/session_test.cpp
    src/domain/session.cpp
)
target_link_libraries(session_test PRIVATE llm_runner_lib GTest::gtest_main)
target_include_directories(session_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(prefix_cache_router_test
    tests/unit/services/prefix_cache_router_test.cpp
    src/domain/session.cpp
    src/domain/prefix_cache/prefix_index.cpp
    src/domain/prefix_cache/block_matcher.cpp
    src/domain/prefix_cache/response_id_index.cpp
    src/services/prefix_cache_router.cpp
    src/metrics/metrics.cpp
)
target_link_libraries(prefix_cache_router_test PRIVATE
    llm_runner_lib Drogon::Drogon prometheus-cpp::core GTest::gtest_main)
target_include_directories(prefix_cache_router_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${prometheus-cpp_SOURCE_DIR}/core/include
)

add_executable(mock_remote_kv_manager_test
    tests/unit/services/mock_remote_kv_manager_test.cpp
    src/services/mock_remote_kv_manager.cpp
)
target_link_libraries(mock_remote_kv_manager_test PRIVATE GTest::gtest_main)
target_include_directories(mock_remote_kv_manager_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(session_manager_test
    tests/integration/sessions/session_manager_test.cpp
    src/domain/session.cpp
    src/domain/prefix_cache/prefix_index.cpp
    src/domain/prefix_cache/block_matcher.cpp
    src/domain/prefix_cache/response_id_index.cpp
    src/services/session_manager.cpp
    src/services/prefix_cache_router.cpp
    src/metrics/metrics.cpp
)
target_link_libraries(session_manager_test PRIVATE
    llm_runner_lib Drogon::Drogon prometheus-cpp::core GTest::gtest_main)
target_include_directories(session_manager_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${prometheus-cpp_SOURCE_DIR}/core/include
)

add_executable(worker_manager_test
    tests/unit/workers/worker_manager_test.cpp
    src/runtime/worker/process_liveness.cpp
)
target_link_libraries(worker_manager_test PRIVATE
    llm_runner_lib GTest::gtest_main)
target_include_directories(worker_manager_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# LLMService unit test: drives processStreamingRequest in-process.
# Pulls in the same translation units that the main server links so the
# real LLMService can be constructed without forking workers.
add_executable(llm_service_test
    tests/integration/sessions/llm_service_test.cpp
    src/services/llm_service.cpp
    src/services/embedding_service.cpp
    src/services/image_service.cpp
    src/services/media_worker_scheduler.cpp
    src/services/model_service_registration.cpp
    src/runtime/worker/worker_manager.cpp
    src/runtime/worker/process_liveness.cpp
    src/runtime/worker/single_process_worker.cpp
    src/runtime/worker/single_process_worker_metrics.cpp
    src/runtime/worker/worker_metrics_shm.cpp
    src/utils/ipc_runner_factory.cpp
    src/utils/mapper.cpp
    src/utils/image_codec.cpp
    src/runtime/runners/embedding_runner.cpp
    src/runtime/runners/sdxl/sdxl_runner.cpp
    src/runtime/runners/media_ipc_runner.cpp
    src/runtime/runners/image_ipc_runner.cpp
    src/runtime/runners/sdxl/sdxl_generate_runner.cpp
    src/runtime/runners/sdxl/sdxl_image_to_image_runner.cpp
    src/runtime/runners/sdxl/sdxl_edit_runner.cpp
    src/runtime/runners/sdxl/sdxl_python_helpers.cpp
    src/metrics/metrics.cpp
    src/profiling/tracy.cpp
)
target_link_libraries(llm_service_test PRIVATE
    llm_runner_lib
    prometheus-cpp::core
    pybind11::embed
    stb_image_write
    GTest::gtest_main)
target_include_directories(llm_service_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${prometheus-cpp_SOURCE_DIR}/core/include
)

# Common server-stack sources shared by every integration-test executable.
# Each integration binary boots the full stack in-process and differs only by
# its test-entry .cpp (and the MODEL it pins), so the source list is shared here
# rather than duplicated per target.
set(INTEGRATION_TEST_COMMON_SOURCES
    src/services/llm_service.cpp
    src/services/embedding_service.cpp
    src/services/model_service_registration.cpp
    src/services/session_manager.cpp
    src/services/prefix_cache_router.cpp
    src/services/session_resolution.cpp
    src/services/memory_services/memory_manager.cpp
    src/domain/session.cpp
    src/domain/prefix_cache/prefix_index.cpp
    src/domain/prefix_cache/block_matcher.cpp
    src/domain/prefix_cache/response_id_index.cpp
    src/runtime/worker/worker_manager.cpp
    src/runtime/worker/process_liveness.cpp
    src/runtime/worker/single_process_worker.cpp
    src/runtime/worker/single_process_worker_metrics.cpp
    src/runtime/worker/worker_metrics_shm.cpp
    src/utils/ipc_runner_factory.cpp
    src/utils/mapper.cpp
    src/utils/conversation_hasher.cpp
    src/utils/service_factory.cpp
    src/runtime/runners/embedding_runner.cpp
    src/runtime/runners/sdxl/sdxl_runner.cpp
    src/runtime/runners/media_ipc_runner.cpp
    src/runtime/runners/image_ipc_runner.cpp
    src/runtime/runners/sdxl/sdxl_generate_runner.cpp
    src/runtime/runners/sdxl/sdxl_image_to_image_runner.cpp
    src/runtime/runners/sdxl/sdxl_edit_runner.cpp
    src/runtime/runners/sdxl/sdxl_python_helpers.cpp
    src/services/image_service.cpp
    src/services/media_worker_scheduler.cpp
    src/utils/image_codec.cpp
    src/metrics/metrics.cpp
    src/profiling/tracy.cpp
    src/api/llm_controller.cpp
    src/api/stream_event_formatter.cpp
    src/api/health_controller.cpp
    src/api/response_writer/response_writer.cpp
    src/api/response_writer/non_stream_response_writer.cpp
    src/api/response_writer/streaming_response_writer.cpp
    src/api/route_registry.cpp
    src/services/service_container.cpp
    src/services/disaggregation_service.cpp
    src/services/llm_pipeline.cpp
    src/sockets/inter_server_service.cpp
    src/sockets/socket_manager.cpp
    src/sockets/zmq_socket_transport.cpp
)

add_executable(main_integration_test
    tests/integration/server/main_integration_test.cpp
    ${INTEGRATION_TEST_COMMON_SOURCES}
)
target_link_libraries(main_integration_test PRIVATE
    llm_runner_lib
    Drogon::Drogon
    cereal::cereal
    prometheus-cpp::core
    pybind11::embed
    stb_image_write
    xxhash_lib
    GTest::gtest_main)
if(ZMQ_FOUND AND TARGET PkgConfig::ZMQ)
    target_link_libraries(main_integration_test PRIVATE PkgConfig::ZMQ)
elseif(TARGET libzmq-static)
    target_link_libraries(main_integration_test PRIVATE libzmq-static)
elseif(TARGET libzmq)
    target_link_libraries(main_integration_test PRIVATE libzmq)
endif()
target_include_directories(main_integration_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${cppzmq_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${prometheus-cpp_SOURCE_DIR}/core/include
)

add_executable(prefill_integration_test
    tests/integration/server/prefill_integration_test.cpp
    ${INTEGRATION_TEST_COMMON_SOURCES}
)
target_link_libraries(prefill_integration_test PRIVATE
    llm_runner_lib
    Drogon::Drogon
    cereal::cereal
    prometheus-cpp::core
    pybind11::embed
    stb_image_write
    xxhash_lib
    GTest::gtest_main)
if(ZMQ_FOUND AND TARGET PkgConfig::ZMQ)
    target_link_libraries(prefill_integration_test PRIVATE PkgConfig::ZMQ)
elseif(TARGET libzmq-static)
    target_link_libraries(prefill_integration_test PRIVATE libzmq-static)
elseif(TARGET libzmq)
    target_link_libraries(prefill_integration_test PRIVATE libzmq)
endif()
target_include_directories(prefill_integration_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${cppzmq_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${prometheus-cpp_SOURCE_DIR}/core/include
)

add_executable(disaggregated_e2e_test
    tests/e2e/disaggregated_e2e_test.cpp
    ${INTEGRATION_TEST_COMMON_SOURCES}
)
target_link_libraries(disaggregated_e2e_test PRIVATE
    llm_runner_lib
    Drogon::Drogon
    cereal::cereal
    prometheus-cpp::core
    pybind11::embed
    stb_image_write
    xxhash_lib
    GTest::gtest_main)
if(ZMQ_FOUND AND TARGET PkgConfig::ZMQ)
    target_link_libraries(disaggregated_e2e_test PRIVATE PkgConfig::ZMQ)
elseif(TARGET libzmq-static)
    target_link_libraries(disaggregated_e2e_test PRIVATE libzmq-static)
elseif(TARGET libzmq)
    target_link_libraries(disaggregated_e2e_test PRIVATE libzmq)
endif()
target_include_directories(disaggregated_e2e_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${cppzmq_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${prometheus-cpp_SOURCE_DIR}/core/include
)

include(GoogleTest)
# main_integration_test boots the whole server stack (services, IPC queues,
# Drogon listener, worker subprocess) once per process via SetUpTestSuite, and
# tears it down via TearDownTestSuite. Registering each TEST_F as its own
# ctest entry would invoke the binary once per test, paying that boot cost six
# times AND exposing a racy teardown. Run all six tests in a single process.
add_test(NAME MainIntegrationTest COMMAND main_integration_test)
add_test(NAME PrefillIntegrationTest COMMAND prefill_integration_test)
add_test(NAME DisaggregatedE2ETest COMMAND disaggregated_e2e_test)
# Only built under Mooncake (TARGET transfer_engine); register its host+builtin
# gtest round trip when present.
if(TARGET transport_kv_migration_e2e)
    add_test(NAME TransportKvMigrationE2E COMMAND transport_kv_migration_e2e)
endif()
# Integration tests boot the full server stack and share ports/resources.
# Prevent parallel execution via RESOURCE_LOCK.
set_tests_properties(MainIntegrationTest PrefillIntegrationTest DisaggregatedE2ETest PROPERTIES
    TIMEOUT 120
    RESOURCE_LOCK integration_server
)
gtest_discover_tests(transport_test)
gtest_discover_tests(dynamo_protocol_test)
gtest_discover_tests(kv_cache_addressing_test)
gtest_discover_tests(kv_table_adapter_test)
gtest_discover_tests(kv_control_channel_test)
gtest_discover_tests(mooncake_kv_migration_test)
gtest_discover_tests(kv_migration_orchestrator_test)
if(ENABLE_KV_TABLE)
    gtest_discover_tests(kv_chunk_address_table_adapter_test)
endif()
if(ENABLE_BLAZE)
    gtest_discover_tests(blaze_decode_runner_test)
    gtest_discover_tests(blaze_prefill_runner_test)
    gtest_discover_tests(mock_schedulers_test)
endif()
gtest_discover_tests(sequence_test)
gtest_discover_tests(lockfree_queue_test)
gtest_discover_tests(tokenizer_test)
gtest_discover_tests(conversation_hasher_test)
gtest_discover_tests(cancel_queue_test)
gtest_discover_tests(structured_output_test)
gtest_discover_tests(worker_metrics_shm_test)
gtest_discover_tests(concurrent_map_test)
gtest_discover_tests(session_test)
gtest_discover_tests(prefix_cache_router_test)
gtest_discover_tests(session_manager_test)
gtest_discover_tests(worker_manager_test)
gtest_discover_tests(llm_service_test)
gtest_discover_tests(model_service_registry_test)
gtest_discover_tests(health_controller_test)
gtest_discover_tests(image_generate_request_test)
gtest_discover_tests(media_payload_ipc_test)

add_executable(apply_delta_prompt_test tests/unit/model/apply_delta_prompt_test.cpp)
target_link_libraries(apply_delta_prompt_test PRIVATE GTest::gtest_main llm_runner_lib)
target_include_directories(apply_delta_prompt_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_test(NAME ApplyDeltaPrompt COMMAND apply_delta_prompt_test)

# Cancellation E2E test (starts mock server, runs Python tests, stops server)
add_test(NAME CancellationE2E
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/scripts/run_e2e_with_server.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(CancellationE2E PROPERTIES TIMEOUT 120)

# Prefix cache E2E test (requires external Dynamo server)
add_executable(prefix_cache_e2e_test tests/e2e/prefix_cache_e2e_test.cpp)
target_link_libraries(prefix_cache_e2e_test PRIVATE
    JsonCpp::JsonCpp
    GTest::gtest_main
)
target_include_directories(prefix_cache_e2e_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Tokenizer benchmark
add_executable(tokenizer_benchmark benchmarks/tokenizer_benchmark.cpp)
target_link_libraries(tokenizer_benchmark PRIVATE llm_runner_lib)

# Prefix cache router microbenchmark (Google Benchmark)
add_executable(prefix_cache_router_benchmark
    benchmarks/prefix_cache_router_benchmark.cpp
    src/domain/session.cpp
    src/domain/prefix_cache/prefix_index.cpp
    src/domain/prefix_cache/block_matcher.cpp
    src/domain/prefix_cache/response_id_index.cpp
    src/services/prefix_cache_router.cpp
    src/metrics/metrics.cpp
)
target_link_libraries(prefix_cache_router_benchmark PRIVATE
    llm_runner_lib Drogon::Drogon prometheus-cpp::core benchmark::benchmark benchmark::benchmark_main)
target_include_directories(prefix_cache_router_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${prometheus-cpp_SOURCE_DIR}/core/include
)
add_test(NAME PrefixCacheRouterBenchmark COMMAND prefix_cache_router_benchmark --benchmark_min_time=0.1s)

# Kafka messaging support (optional)
if(KAFKA_ENABLED)
    # Kafka messaging library
    find_library(RDKAFKA_LIB rdkafka REQUIRED)
    find_library(RDKAFKACPP_LIB rdkafka++ REQUIRED)
    find_path(RDKAFKA_INCLUDE_DIR librdkafka/rdkafka.h REQUIRED)

    message(STATUS "Kafka support enabled")
    message(STATUS "  librdkafka: ${RDKAFKA_LIB}")
    message(STATUS "  librdkafka++: ${RDKAFKACPP_LIB}")
    message(STATUS "  include: ${RDKAFKA_INCLUDE_DIR}")

    add_library(messaging
        src/messaging/kafka_producer.cpp
        src/messaging/kafka_consumer.cpp
        src/messaging/migration_message.cpp
        src/services/remote_kv_manager_impl.cpp
        src/utils/logger.cpp
    )
    target_include_directories(messaging PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${RDKAFKA_INCLUDE_DIR}
    )
    target_link_libraries(messaging PUBLIC
        ${RDKAFKA_LIB}
        ${RDKAFKACPP_LIB}
        spdlog::spdlog
        JsonCpp::JsonCpp
    )

    # RemoteKVManagerAdapter - bridges IRemoteKVManager to MigrationClientInterface
    # Only built when ENABLE_BLAZE is ON (requires tt-llm-engine headers)
    # Note: disaggregation/migration/include is needed for migration_client.hpp
    # which migration_client_interface.hpp includes for event types.
    if(ENABLE_BLAZE)
        add_library(remote_kv_adapter
            src/services/remote_kv_manager_adapter.cpp
        )
        target_include_directories(remote_kv_adapter PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/tt-llm-engine/include
            ${CMAKE_CURRENT_SOURCE_DIR}/tt-llm-engine/disaggregation/migration/include
        )
        target_link_libraries(remote_kv_adapter PUBLIC
            messaging
            spdlog::spdlog
        )
        message(STATUS "remote_kv_adapter: ENABLED (ENABLE_BLAZE=ON)")
    else()
        message(STATUS "remote_kv_adapter: DISABLED (requires ENABLE_BLAZE=ON)")
    endif()

    # Migration worker library (for consumer instances)
    add_library(migration_worker
        src/runtime/worker/migration_worker.cpp
    )
    target_include_directories(migration_worker PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(migration_worker PUBLIC
        messaging
        spdlog::spdlog
        JsonCpp::JsonCpp
    )
    # Unit test for migration_message JSON (de)serialization.
    add_executable(migration_message_test
        tests/unit/messaging/migration_message_test.cpp
    )
    target_link_libraries(migration_message_test PRIVATE
        messaging
        GTest::gtest_main
    )
    target_include_directories(migration_message_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Unit test for RemoteKVManagerImpl (uses in-process fake producer/consumer).
    add_executable(remote_kv_manager_impl_test
        tests/unit/services/remote_kv_manager_impl_test.cpp
    )
    target_link_libraries(remote_kv_manager_impl_test PRIVATE
        messaging
        GTest::gtest_main
    )
    target_include_directories(remote_kv_manager_impl_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # KV migration worker library
    add_library(kv_migration_worker
        src/runtime/worker/kv_migration_worker.cpp
        src/runtime/worker/stub_migration_executor.cpp
    )
    target_include_directories(kv_migration_worker PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(kv_migration_worker PUBLIC
        messaging
        spdlog::spdlog
    )

    # #4294 production worker entry point: combines Mooncake bring-up
    # (transport_lib) with the Kafka request/ack loop (kv_migration_worker).
    # Gated on `transfer_engine` since the Mooncake transport is what makes
    # the bring-up meaningful; without it the binary would have nothing to
    # publish/discover and would only exercise the stub executor path.
    if(TARGET transfer_engine)
        # Links `messaging` (Kafka clients + migration_message parsers) rather
        # than the heavier `kv_migration_worker` lib: the Kafka loop is inlined
        # in bringup_mooncake_worker.cpp on purpose so MooncakeMigrationWorker
        # stays Kafka-blind and this binary doesn't share its consumer-loop
        # body with src/kv_migration_consumer_main.cpp.
        add_executable(bringup_mooncake_worker
            src/transport/bringup_mooncake_worker.cpp)
        target_link_libraries(bringup_mooncake_worker PRIVATE
            transport_lib
            messaging)
        target_include_directories(bringup_mooncake_worker PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include)
        message(STATUS "bringup_mooncake_worker: ENABLED (#4294 production worker)")

        # MPI discovery e2e test: launches 4 prefill + 16 decode
        # bringup_mooncake_worker processes on one host and checks every worker
        # discovers its peers. Registered here so the bringup_mooncake_worker
        # target is guaranteed to exist; skipped when mpirun/mpiexec is missing.
        # MooncakeKafkaMigration extends this with the Kafka request/ack
        # path; it needs the dev Kafka broker running (scripts/dev-kafka.sh up)
        # and topics provisioned (scripts/migration_cli.py setup), so it is
        # skipped automatically when those aren't available.
        find_program(MPIEXEC_EXECUTABLE NAMES mpirun mpiexec)
        if(MPIEXEC_EXECUTABLE)
            add_test(NAME MooncakeMpiDiscovery
                COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/scripts/run_migration_workers_mpi.sh
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
            set_tests_properties(MooncakeMpiDiscovery PROPERTIES
                TIMEOUT 180
                RESOURCE_LOCK integration_server
                ENVIRONMENT "WORKER_BIN=$<TARGET_FILE:bringup_mooncake_worker>")
            message(STATUS "MooncakeMpiDiscovery: ENABLED (#4294 MPI e2e test)")

            add_test(NAME MooncakeKafkaMigration
                COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/e2e/scripts/run_kafka_migration_workers_mpi.py
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
            set_tests_properties(MooncakeKafkaMigration PROPERTIES
                TIMEOUT 240
                RESOURCE_LOCK integration_server
                ENVIRONMENT "WORKER_BIN=$<TARGET_FILE:bringup_mooncake_worker>")
            message(STATUS "MooncakeKafkaMigration: ENABLED (4 prefill + 16 decode, end-to-end Kafka)")
        else()
            message(STATUS "MooncakeMpiDiscovery: SKIPPED (mpirun/mpiexec not found)")
            message(STATUS "MooncakeKafkaMigration: SKIPPED (mpirun/mpiexec not found)")
        endif()
    else()
        message(STATUS "bringup_mooncake_worker: SKIPPED (requires Mooncake transfer_engine target)")
    endif()

    # Unit test for KvMigrationWorker (uses in-process fake Kafka + mock executor).
    add_executable(kv_migration_worker_test
        tests/unit/workers/kv_migration_worker_test.cpp
    )
    target_link_libraries(kv_migration_worker_test PRIVATE
        kv_migration_worker
        messaging
        GTest::gtest_main
    )
    target_include_directories(kv_migration_worker_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    add_executable(remote_kv_manager_kafka_test
        tests/integration/remote_kv_manager_kafka_test.cpp
    )
    target_link_libraries(remote_kv_manager_kafka_test PRIVATE
        kv_migration_worker
        messaging
        GTest::gtest_main
    )
    target_include_directories(remote_kv_manager_kafka_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # E2E test: real RemoteKVManagerImpl driven against a scriptable
    # MockMigrationWorker over a real Kafka broker (scripts/dev-kafka.sh up).
    # Not registered with add_test() -- opt-in, run manually.
    add_executable(remote_kv_manager_e2e_test
        tests/e2e/remote_kv_manager_e2e_test.cpp
    )
    target_link_libraries(remote_kv_manager_e2e_test PRIVATE
        messaging
        GTest::gtest_main
    )
    target_include_directories(remote_kv_manager_e2e_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
else()
    message(STATUS "Kafka support disabled (use -DKAFKA_ENABLED=ON to enable)")
endif()

# Source files that aren't provided by llm_runner_lib
set(SOURCES
    src/main.cpp
    src/profiling/tracy.cpp
    src/api/llm_controller.cpp
    src/api/stream_event_formatter.cpp
    src/api/response_writer/response_writer.cpp
    src/api/response_writer/streaming_response_writer.cpp
    src/api/response_writer/non_stream_response_writer.cpp
    src/api/health_controller.cpp
    src/api/info_controller.cpp
    src/api/openapi.cpp
    src/services/llm_service.cpp
    src/domain/prefix_cache/prefix_index.cpp
    src/domain/prefix_cache/block_matcher.cpp
    src/domain/prefix_cache/response_id_index.cpp
    src/services/session_manager.cpp
    src/services/prefix_cache_router.cpp
    src/services/session_resolution.cpp
    src/domain/session.cpp
    src/runtime/worker/single_process_worker.cpp
    src/runtime/worker/worker_manager.cpp
    src/runtime/worker/process_liveness.cpp
    src/utils/service_factory.cpp
    src/services/service_container.cpp
    src/services/model_service_registration.cpp
    src/utils/ipc_runner_factory.cpp
    src/utils/mapper.cpp
    src/api/embedding_controller.cpp
    src/runtime/runners/embedding_runner.cpp
    src/services/embedding_service.cpp
    src/api/image_controller.cpp
    src/runtime/runners/sdxl/sdxl_runner.cpp
    src/runtime/runners/media_ipc_runner.cpp
    src/runtime/runners/image_ipc_runner.cpp
    src/runtime/runners/sdxl/sdxl_generate_runner.cpp
    src/runtime/runners/sdxl/sdxl_image_to_image_runner.cpp
    src/runtime/runners/sdxl/sdxl_edit_runner.cpp
    src/runtime/runners/sdxl/sdxl_python_helpers.cpp
    src/services/image_service.cpp
    src/services/media_worker_scheduler.cpp
    src/utils/image_codec.cpp
    src/services/disaggregation_service.cpp
    src/services/llm_pipeline.cpp
    src/sockets/socket_manager.cpp
    src/sockets/zmq_socket_transport.cpp
    src/sockets/inter_server_service.cpp
    src/metrics/metrics.cpp
    src/api/metrics_controller.cpp
    src/runtime/worker/single_process_worker_metrics.cpp
    src/runtime/worker/worker_metrics_shm.cpp
    src/runtime/worker/worker_metrics_aggregator.cpp
    src/runtime/worker/blaze_worker_metrics_renderer.cpp
    src/dynamo/dynamo_protocol.cpp
    src/dynamo/discovery.cpp
    src/dynamo/etcd_client.cpp
    src/dynamo/dynamo_endpoint.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/generated
    ${rapidjson_SOURCE_DIR}/include
    ${TT_METAL_INCLUDE_DIRS}
    ${prometheus-cpp_SOURCE_DIR}/core/include
)

# Build-time codegen for config/build_info.hpp (build-identity constants
# consumed by InfoController and any future caller that wants to label a
# log line / metric / response with the running binary's version + SHAs).
# Runs on every build so commit SHAs reflect the current tree, not whatever
# was checked out at the last `cmake -B build` invocation.
add_custom_target(generate_build_info ALL
    COMMAND ${CMAKE_COMMAND}
        -DCPP_SERVER_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DTEMPLATE=${CMAKE_CURRENT_SOURCE_DIR}/include/config/build_info.hpp.in
        -DOUTPUT=${CMAKE_BINARY_DIR}/generated/config/build_info.hpp
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateBuildInfo.cmake
    BYPRODUCTS ${CMAKE_BINARY_DIR}/generated/config/build_info.hpp
    COMMENT "Generating config/build_info.hpp from VERSION + git"
    VERBATIM
)
add_dependencies(${PROJECT_NAME} generate_build_info)

# Link Drogon, llm_runner_lib, and prometheus-cpp core
if(USE_EXTERNAL_FMT)
    target_link_libraries(${PROJECT_NAME} PRIVATE Drogon::Drogon llm_runner_lib cereal::cereal spdlog::spdlog fmt::fmt prometheus-cpp::core)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE Drogon::Drogon llm_runner_lib cereal::cereal spdlog::spdlog prometheus-cpp::core)
endif()

# Link ZeroMQ
if(ZMQ_FOUND AND TARGET PkgConfig::ZMQ)
    target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::ZMQ)
elseif(TARGET libzmq-static)
    target_link_libraries(${PROJECT_NAME} PRIVATE libzmq-static)
elseif(TARGET libzmq)
    target_link_libraries(${PROJECT_NAME} PRIVATE libzmq)
endif()
target_include_directories(${PROJECT_NAME} PRIVATE ${cppzmq_SOURCE_DIR})

# Link Kafka messaging if enabled
if(KAFKA_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE messaging)
endif()
if(ENABLE_TRACY)
    target_sources(${PROJECT_NAME} PRIVATE src/profiling/tracy_alloc.cpp)
    target_compile_definitions(${PROJECT_NAME} PRIVATE TT_ENABLE_TRACY)
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::embed)

# Tokenizer (requires Rust) and tokenizer_config (JsonCpp)
target_link_libraries(${PROJECT_NAME} PRIVATE tokenizers_cpp JsonCpp::JsonCpp)

# BLAKE3 for Dynamo MDC file checksums (src/dynamo/discovery.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE blake3)

target_link_libraries(${PROJECT_NAME} PRIVATE stb_image_write)

# Make the executable's runtime library lookup self-contained relative to
# its own location (DT_RUNPATH = $ORIGIN/...). Required so the build
# artifact can be unpacked and run on a different machine than the
# build runner (e.g. CI build on ubuntu-latest, bench on a self-hosted
# runner with a different workspace path). Without this, CMake bakes the
# absolute build-time path into RUNPATH and ld.so fails to find
# libtt_llm_engine_core.so.0 elsewhere.
if(ENABLE_BLAZE)
    # $ORIGIN/tt-llm-engine keeps libtt_llm_engine_core.so.0 / libtt_llm_engine.so.0
    # resolvable when the build artifact is unpacked on a different machine.
    # INSTALL_RPATH_USE_LINK_PATH TRUE additionally appends the absolute link
    # directories (e.g. tt-llm-engine/tt-metal/build_Release/lib where libfmt.so.11,
    # libtt_metal.so, libtt_stl.so live) so local dev runs don't need a manual
    # LD_LIBRARY_PATH. Those absolute paths are harmless for the CI case since
    # they just won't resolve on a different runner, while $ORIGIN still does.
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN/tt-llm-engine"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
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


# Consumer executable (separate Drogon instance for Kafka consumption)
if(KAFKA_ENABLED)
    add_executable(tt_consumer
        src/consumer_main.cpp
    )
    target_include_directories(tt_consumer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(tt_consumer PRIVATE
        Drogon::Drogon
        llm_runner_lib
        migration_worker
        messaging
    )
    if(USE_EXTERNAL_FMT)
        target_link_libraries(tt_consumer PRIVATE spdlog::spdlog fmt::fmt)
    else()
        target_link_libraries(tt_consumer PRIVATE spdlog::spdlog)
    endif()
    target_compile_options(tt_consumer PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )

    # Separate process so the legacy `tt_consumer` is untouched.
    add_executable(tt_kv_migration_consumer
        src/kv_migration_consumer_main.cpp
    )
    target_include_directories(tt_kv_migration_consumer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(tt_kv_migration_consumer PRIVATE
        Drogon::Drogon
        llm_runner_lib
        kv_migration_worker
        messaging
    )
    if(USE_EXTERNAL_FMT)
        target_link_libraries(tt_kv_migration_consumer PRIVATE
            spdlog::spdlog fmt::fmt)
    else()
        target_link_libraries(tt_kv_migration_consumer PRIVATE spdlog::spdlog)
    endif()
    target_compile_options(tt_kv_migration_consumer PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# AddressSanitizer for executables (must be after all add_executable calls)
if(SANITIZE_ADDRESS)
    set(ASAN_FLAGS -fsanitize=address -fno-omit-frame-pointer -g)
    set(ASAN_TARGETS
        ${PROJECT_NAME}
        sequence_test
        lockfree_queue_test
        cancel_queue_test
        structured_output_test
    )
    if(KAFKA_ENABLED)
        list(APPEND ASAN_TARGETS tt_consumer)
    endif()
    foreach(target IN LISTS ASAN_TARGETS)
        target_compile_options(${target} PRIVATE ${ASAN_FLAGS})
        target_link_options(${target} PRIVATE -fsanitize=address)
    endforeach()
endif()
