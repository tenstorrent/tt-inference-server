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
#include <vector>

#include "config/types.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::utils::tts_tokenizer {

inline constexpr const char* SPEECH_END_TOKEN = "<|speech_end|>";
inline constexpr const char* SPEECH_TOKEN_BASE = "<|s_0|>";
inline constexpr const char* SPEECH_START_TOKEN = "<|speech_start|>";
inline constexpr const char* SPEECH_TOKEN_NEXT = "<|s_1|>";
inline constexpr const char* SPEECH_TOKEN_PATTERN_PREFIX = "<|s_";
inline constexpr const char* SPEECH_TOKEN_PATTERN_SUFFIX = "|>";
inline constexpr const char* AUDIO_PROMPT_START_TOKEN =
    "<|audio_prompt_start|>";
inline constexpr const char* AUDIO_PROMPT_END_TOKEN = "<|audio_prompt_end|>";
inline constexpr const char* VOICE_PROMPT_START_TOKEN =
    "<|voice_prompt_start|>";
inline constexpr const char* VOICE_PROMPT_END_TOKEN = "<|voice_prompt_end|>";
inline constexpr const char* BOT_TOKEN = "<|bot|>";

inline std::string speechTokenForId(uint32_t speechId) {
  return std::string(SPEECH_TOKEN_PATTERN_PREFIX) + std::to_string(speechId) +
         SPEECH_TOKEN_PATTERN_SUFFIX;
}

inline uint32_t tokenIdForVocab(const std::vector<std::string>& vocab,
                                const std::string& token) {
  auto it = std::find(vocab.begin(), vocab.end(), token);
  if (it == vocab.end()) {
    throw std::runtime_error("TTS tokenizer is missing required token: " +
                             token);
  }
  return static_cast<uint32_t>(std::distance(vocab.begin(), it));
}

inline void validateRequiredTokens(const tokenizers::Tokenizer& tokenizer) {
  const auto vocab = tokenizer.getEncodedVocab();
  for (const char* token :
       {SPEECH_START_TOKEN, SPEECH_END_TOKEN, SPEECH_TOKEN_BASE,
        SPEECH_TOKEN_NEXT, AUDIO_PROMPT_START_TOKEN, AUDIO_PROMPT_END_TOKEN,
        VOICE_PROMPT_START_TOKEN, VOICE_PROMPT_END_TOKEN, BOT_TOKEN}) {
    tokenIdForVocab(vocab, token);
  }
}

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
    validateRequiredTokens(*tokenizer);
  }
  return *tokenizer;
}

inline uint32_t tokenIdFor(const tokenizers::Tokenizer& tokenizer,
                           const std::string& token) {
  const auto vocab = tokenizer.getEncodedVocab();
  return tokenIdForVocab(vocab, token);
}

}  // namespace tt::utils::tts_tokenizer
