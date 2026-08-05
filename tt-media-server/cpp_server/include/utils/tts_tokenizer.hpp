// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "config/types.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::utils::tts_tokenizer {

inline constexpr const char* SPEECH_END_TOKEN = "<|speech_end|>";
inline constexpr const char* SPEECH_TOKEN_BASE = "<|s_0|>";

inline const tokenizers::Tokenizer& tokenizerForPath(
    const std::string& tokenizerPath) {
  if (tokenizerPath.empty()) {
    throw std::runtime_error(
        "TTS tokenizer path is empty; set TTS_TOKENIZER_PATH");
  }

  thread_local std::unordered_map<std::string,
                                  std::unique_ptr<tokenizers::Tokenizer>>
      tokenizersByPath;
  auto& tokenizer = tokenizersByPath[tokenizerPath];
  if (!tokenizer) {
    tokenizer = tokenizers::createTokenizer(
        config::ModelType::LLAMA_3_1_8B_INSTRUCT, tokenizerPath);
  }
  return *tokenizer;
}

inline uint32_t tokenIdFor(const tokenizers::Tokenizer& tokenizer,
                           const std::string& token) {
  const auto vocab = tokenizer.getEncodedVocab();
  auto it = std::find(vocab.begin(), vocab.end(), token);
  if (it == vocab.end()) {
    throw std::runtime_error("TTS tokenizer is missing required token: " +
                             token);
  }
  return static_cast<uint32_t>(std::distance(vocab.begin(), it));
}

}  // namespace tt::utils::tts_tokenizer
