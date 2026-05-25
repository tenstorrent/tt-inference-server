# SPDX-License-Identifier: Apache-2.0
# Patches tokenizers-cpp so SentencePiece is not built (avoids protobuf clash with gRPC).
# FetchContent runs this with the tokenizers-cpp source directory as the working directory.
cmake_minimum_required(VERSION 3.25)

set(CMAKELISTS "CMakeLists.txt")
if(NOT EXISTS "${CMAKELISTS}")
    message(FATAL_ERROR "PatchTokenizersForGrpc: ${CMAKELISTS} not found in ${CMAKE_CURRENT_LIST_DIR}")
endif()

file(READ "${CMAKELISTS}" CONTENT)

string(REPLACE
"set(
  TOKENIZER_CPP_SRCS
  src/sentencepiece_tokenizer.cc
  src/huggingface_tokenizer.cc
  src/rwkv_world_tokenizer.cc
)
add_library(tokenizers_cpp STATIC \${TOKENIZER_CPP_SRCS})
target_include_directories(tokenizers_cpp PRIVATE sentencepiece/src)"
"set(TOKENIZER_CPP_SRCS
  src/huggingface_tokenizer.cc
  src/rwkv_world_tokenizer.cc
)
if (MLC_ENABLE_SENTENCEPIECE_TOKENIZER STREQUAL \"ON\")
  list(APPEND TOKENIZER_CPP_SRCS src/sentencepiece_tokenizer.cc)
endif()
add_library(tokenizers_cpp STATIC \${TOKENIZER_CPP_SRCS})
if (MLC_ENABLE_SENTENCEPIECE_TOKENIZER STREQUAL \"ON\")
  target_include_directories(tokenizers_cpp PRIVATE sentencepiece/src)
endif()"
CONTENT "${CONTENT}")

string(REPLACE
"add_subdirectory(sentencepiece sentencepiece EXCLUDE_FROM_ALL)"
"if (MLC_ENABLE_SENTENCEPIECE_TOKENIZER STREQUAL \"ON\")
  add_subdirectory(sentencepiece sentencepiece EXCLUDE_FROM_ALL)
endif()"
CONTENT "${CONTENT}")

string(REPLACE
"target_link_libraries(tokenizers_cpp PRIVATE tokenizers_c sentencepiece-static \${TOKENIZERS_CPP_LINK_LIBS})"
"if (MLC_ENABLE_SENTENCEPIECE_TOKENIZER STREQUAL \"ON\")
  target_link_libraries(tokenizers_cpp PRIVATE tokenizers_c sentencepiece-static \${TOKENIZERS_CPP_LINK_LIBS})
else()
  target_link_libraries(tokenizers_cpp PRIVATE tokenizers_c \${TOKENIZERS_CPP_LINK_LIBS})
endif()"
CONTENT "${CONTENT}")

file(WRITE "${CMAKELISTS}" "${CONTENT}")
message(STATUS "Patched tokenizers-cpp for gRPC (SentencePiece disabled)")
