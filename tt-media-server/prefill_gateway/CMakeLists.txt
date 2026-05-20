# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# PrefillGateway: stateless dispatcher between 1 decode and N prefill servers.
#
# This file is intentionally lean. The gateway has no LLM engine, no Drogon,
# no runners. It links against cereal (wire format) + spdlog (logging) +
# gtest (unit tests), plus a small slice of cpp_server source for the
# socket-framing layer.

cmake_minimum_required(VERSION 3.19)
project(tt_prefill_gateway CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-g)

if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5
        CACHE STRING "Min policy version for included configs")
endif()

# Path to cpp_server's headers + a small subset of its sources.
# The gateway reuses the existing wire types (socket_messages.hpp) and
# socket framing (SocketManager).
set(CPP_SERVER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cpp_server"
    CACHE PATH "Path to cpp_server source tree")

if(NOT EXISTS "${CPP_SERVER_DIR}/include/sockets/socket_messages.hpp")
    message(FATAL_ERROR
        "Could not find cpp_server at ${CPP_SERVER_DIR}. "
        "Override with -DCPP_SERVER_DIR=/abs/path/to/cpp_server")
endif()

include(FetchContent)

# cereal: binary serialization for the wire protocol.
find_package(cereal QUIET)
if(NOT cereal_FOUND)
    FetchContent_Declare(
        cereal
        GIT_REPOSITORY https://github.com/USCiLab/cereal.git
        GIT_TAG v1.3.2
    )
    set(BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_SANDBOX OFF CACHE BOOL "" FORCE)
    set(SKIP_PERFORMANCE_COMPARISON ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(cereal)
endif()

# spdlog: matches cpp_server's logger; gateway reuses utils/logger.hpp.
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spdlog)

# GoogleTest for policy / dispatcher unit tests.
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
foreach(_gt_target gtest gtest_main gmock gmock_main)
    if(TARGET ${_gt_target})
        target_compile_options(${_gt_target} PRIVATE -Wno-deprecated-declarations)
    endif()
endforeach()

# Gateway core library: selector + registry + affinity cache + dispatcher.
# All pure C++; no socket I/O is performed inside the library. The dispatcher
# receives Senders (std::function objects) which production main() wires to
# real SocketManager instances and tests wire to capture-only lambdas.
set(GATEWAY_CORE_SOURCES
    src/prefill_selector.cpp
    src/affinity_cache.cpp
    src/prefill_registry.cpp
    src/dispatcher.cpp
    ${CPP_SERVER_DIR}/src/utils/logger.cpp
)

add_library(prefill_gateway_core STATIC ${GATEWAY_CORE_SOURCES})
target_include_directories(prefill_gateway_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CPP_SERVER_DIR}/include
)
target_link_libraries(prefill_gateway_core PUBLIC
    cereal::cereal
    spdlog::spdlog
)
target_compile_features(prefill_gateway_core PUBLIC cxx_std_20)
target_compile_options(prefill_gateway_core PRIVATE -Wall -Wextra -Wpedantic)

# libzmq + cppzmq for the ZMQ transport. Mirror cpp_server's strategy:
# prefer system libzmq via pkg-config, fall back to FetchContent.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ IMPORTED_TARGET libzmq)
endif()
if(NOT ZMQ_FOUND)
    find_package(ZeroMQ QUIET)
    if(ZeroMQ_FOUND AND TARGET libzmq)
        set(ZMQ_FOUND TRUE)
    endif()
endif()
if(NOT ZMQ_FOUND)
    message(STATUS "[gateway] libzmq not found; building from source via FetchContent")
    FetchContent_Declare(
        libzmq
        GIT_REPOSITORY https://github.com/zeromq/libzmq.git
        GIT_TAG v4.3.5
    )
    set(ZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(WITH_PERF_TOOL OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(BUILD_STATIC ON CACHE BOOL "" FORCE)
    set(ENABLE_CPACK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(libzmq)
endif()
FetchContent_Declare(
    cppzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG v4.11.0
    GIT_SHALLOW TRUE
)
set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cppzmq)

# Socket + logger sources pulled from cpp_server — shared implementation so
# the gateway uses exactly the same wire format and logger as the inference
# servers, without introducing a second copy of the code. The gateway ships
# its own factory (src/socket_transport_factory.cpp) so that it can honor
# SOCKET_TRANSPORT without pulling cpp_server's settings.cpp (and its
# tokenizer/runner deps).
add_library(prefill_gateway_io STATIC
    ${CPP_SERVER_DIR}/src/sockets/socket_manager.cpp
    ${CPP_SERVER_DIR}/src/sockets/tcp_socket_transport.cpp
    ${CPP_SERVER_DIR}/src/sockets/zmq_socket_transport.cpp
    src/socket_transport_factory.cpp
)
target_include_directories(prefill_gateway_io PUBLIC
    ${CPP_SERVER_DIR}/include
    ${cppzmq_SOURCE_DIR}
)
target_link_libraries(prefill_gateway_io PUBLIC
    cereal::cereal
    spdlog::spdlog
)
if(ZMQ_FOUND AND TARGET PkgConfig::ZMQ)
    target_link_libraries(prefill_gateway_io PUBLIC PkgConfig::ZMQ)
elseif(TARGET libzmq-static)
    target_link_libraries(prefill_gateway_io PUBLIC libzmq-static)
elseif(TARGET libzmq)
    target_link_libraries(prefill_gateway_io PUBLIC libzmq)
endif()
target_compile_features(prefill_gateway_io PUBLIC cxx_std_20)

# prefill_gateway binary — wires core library to real SocketManagers.
add_executable(prefill_gateway src/main.cpp)
target_include_directories(prefill_gateway PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(prefill_gateway PRIVATE
    prefill_gateway_core
    prefill_gateway_io
)
target_compile_features(prefill_gateway PUBLIC cxx_std_20)
target_compile_options(prefill_gateway PRIVATE -Wall -Wextra -Wpedantic)

# Unit tests.
enable_testing()
include(GoogleTest)

# Pure unit tests: depend on prefill_gateway_core only (no socket I/O).
function(_gateway_add_unit_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE
        prefill_gateway_core GTest::gtest_main)
    gtest_discover_tests(${name})
endfunction()

# Integration / e2e tests: need real sockets, so also link prefill_gateway_io.
function(_gateway_add_e2e_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE
        prefill_gateway_core prefill_gateway_io GTest::gtest_main)
    target_include_directories(${name} PRIVATE ${CPP_SERVER_DIR}/include)
    gtest_discover_tests(${name})
endfunction()

_gateway_add_unit_test(prefill_selector_test)
_gateway_add_unit_test(affinity_cache_test)
_gateway_add_unit_test(prefill_registry_test)
_gateway_add_unit_test(dispatcher_test)
_gateway_add_e2e_test(gateway_e2e_test)
