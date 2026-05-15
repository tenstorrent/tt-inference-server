# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Renders include/config/build_info.hpp.in into ${OUTPUT} with build-time
# values for tt-inference-server, tt-llm-engine, and tt-metal versions/commits.
#
# Invoked at build time (not configure time) by an add_custom_target in the
# top-level CMakeLists.txt so commit SHAs stay accurate across incremental
# builds without requiring a CMake reconfigure.
#
# Required variables (passed via -D from the parent CMake):
#   CPP_SERVER_DIR  Absolute path to cpp_server/
#   TEMPLATE        Absolute path to build_info.hpp.in
#   OUTPUT          Absolute path where the rendered header should be written

if(NOT DEFINED CPP_SERVER_DIR OR NOT DEFINED TEMPLATE OR NOT DEFINED OUTPUT)
    message(FATAL_ERROR "GenerateBuildInfo.cmake requires -DCPP_SERVER_DIR -DTEMPLATE -DOUTPUT")
endif()

find_package(Git QUIET)

# Resolve a commit SHA from a directory's .git tree, or set to "unknown".
function(_resolve_git_commit dir_path out_var)
    set(${out_var} "unknown" PARENT_SCOPE)
    if(NOT GIT_FOUND)
        return()
    endif()
    if(NOT EXISTS "${dir_path}/.git")
        return()
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C "${dir_path}" rev-parse HEAD
        OUTPUT_VARIABLE sha
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE rc
    )
    if(rc EQUAL 0 AND NOT sha STREQUAL "")
        set(${out_var} "${sha}" PARENT_SCOPE)
    endif()
endfunction()

# tt-inference-server version: read top-level VERSION file
set(TT_INFERENCE_SERVER_VERSION "unknown")
set(_version_file "${CPP_SERVER_DIR}/../../VERSION")
if(EXISTS "${_version_file}")
    file(READ "${_version_file}" _version_raw)
    string(STRIP "${_version_raw}" TT_INFERENCE_SERVER_VERSION)
endif()

# tt-inference-server commit: passed in by the docker build via
# --build-arg TT_INFERENCE_SERVER_COMMIT_SHA_OR_TAG=$(git rev-parse HEAD).
# The inference-server's .git is not available at build time inside the
# image (only tt-media-server/ is COPY'd), so the env var is the sole
# source — no git fallback. Empty/missing -> "unknown".
set(TT_INFERENCE_SERVER_COMMIT "$ENV{TT_INFERENCE_SERVER_COMMIT_SHA_OR_TAG}")
if(TT_INFERENCE_SERVER_COMMIT STREQUAL "")
    set(TT_INFERENCE_SERVER_COMMIT "unknown")
endif()

# tt-llm-engine commit (exposed in /info as `tt_blaze.commit` for backwards
# compatibility — keep the C++ var name TT_BLAZE_COMMIT in sync with
# build_info.hpp.in / kTtBlazeCommit). git rev-parse on the cloned
# tt-llm-engine working tree. Present at this path inside Dockerfile.blaze
# (cloned at build time); absent in standalone cpp_server builds — falls
# through to "unknown".
_resolve_git_commit("${CPP_SERVER_DIR}/tt-llm-engine" TT_BLAZE_COMMIT)

# tt-metal commit: git rev-parse on tt-llm-engine's tt-metal submodule working tree.
# Intentionally NOT reading $ENV{TT_METAL_COMMIT_SHA_OR_TAG} — that ARG in
# Dockerfile.blaze will be removed; tt-metal is pulled in as a
# submodule of tt-llm-engine (cloned by branch).
_resolve_git_commit("${CPP_SERVER_DIR}/tt-llm-engine/tt-metal" TT_METAL_COMMIT)

# Render the header. configure_file with @ONLY substitutes only @VAR@ tokens
# (not bare ${VAR}), keeping the template robust to copy/paste of macro-heavy
# C++ code if the template ever grows.
configure_file("${TEMPLATE}" "${OUTPUT}" @ONLY)

message(STATUS "build_info.hpp generated:")
message(STATUS "  tt_inference_server.version = ${TT_INFERENCE_SERVER_VERSION}")
message(STATUS "  tt_inference_server.commit  = ${TT_INFERENCE_SERVER_COMMIT}")
message(STATUS "  tt_blaze.commit             = ${TT_BLAZE_COMMIT}")
message(STATUS "  tt_metal.commit             = ${TT_METAL_COMMIT}")
