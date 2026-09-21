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
inline constexpr const char* USER_TOKEN = "<|user|>";
// Instruction wrappers. TTS-2 is trained with wrap_instructions enabled, so a
// turn instruction goes BETWEEN these rather than in a bracketed
// "[say with anger] ..." prefix. NEW in the 2026-09 vocabulary: they replaced
// <|reserved_token_68|>/<|reserved_token_69|>, so an older checkpoint does not
// have them and would silently tokenize the literal as ordinary text.
inline constexpr const char* INSTRUCTION_START_TOKEN = "<|instruction_start|>";
inline constexpr const char* INSTRUCTION_END_TOKEN = "<|instruction_end|>";
// Readout tokens: reserved tokens the model was trained to see immediately
// before EVERY generated speech segment.
inline constexpr uint32_t READOUT_TOKEN_COUNT = 8;
inline std::string readoutTokens() {
  std::string out;
  for (uint32_t i = 0; i < READOUT_TOKEN_COUNT; ++i) {
    out += "<|reserved_token_" + std::to_string(i) + "|>";
  }
  return out;
}

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
        VOICE_PROMPT_START_TOKEN, VOICE_PROMPT_END_TOKEN, BOT_TOKEN,
        // v2 prompt format. Requiring these is also the cheapest guard against
        // a pre-2026-09 checkpoint: that vocabulary renumbered every token and
        // has no instruction wrappers at all, which otherwise fails silently.
        INSTRUCTION_START_TOKEN, INSTRUCTION_END_TOKEN,
        "<|reserved_token_0|>", "<|reserved_token_7|>"}) {
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
