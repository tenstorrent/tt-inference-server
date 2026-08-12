// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <algorithm>
#include <cctype>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/tokenizers/tokenizer.hpp"
#include "utils/tts_tokenizer.hpp"

namespace tt::utils::tts_prompt_compiler {

namespace tts_tokens = tt::utils::tts_tokenizer;

// TTS-2 prompt format:
//   TVD / description-only:
//     <|voice_prompt_start|>{description}<|voice_prompt_end|>
//     <|bot|>{text}<|speech_start|>
//   Voice-clone continuation:
//     <|audio_prompt_start|><|s_12|><|s_34|>...<|audio_prompt_end|>
//     <|voice_prompt_start|>{description}<|voice_prompt_end|>
//     <|bot|>{text}<|speech_start|>
//
// The final string is tokenized by the TTS tokenizer; speech IDs are encoded as
// literal tokenizer tokens like <|s_123|>, not inserted as raw token IDs.
inline std::string trim(std::string value) {
  auto isNotSpace = [](unsigned char ch) { return std::isspace(ch) == 0; };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), isNotSpace));
  value.erase(std::find_if(value.rbegin(), value.rend(), isNotSpace).base(),
              value.end());
  return value;
}

inline void appendSpeechTokens(std::ostringstream& prompt,
                               const std::vector<uint32_t>& speechIds) {
  for (uint32_t speechId : speechIds) {
    prompt << tts_tokens::speechTokenForId(speechId);
  }
}

inline void validatePromptInputs(
    const std::string& text, const std::optional<std::string>& description) {
  const std::string trimmedText = trim(text);
  if (trimmedText.empty()) {
    throw std::invalid_argument("TTS text must not be empty");
  }
  if (description.has_value()) {
    const std::string trimmedDescription = trim(*description);
    if (trimmedDescription.empty()) {
      throw std::invalid_argument(
          "TTS voice description must not be empty when provided");
    }
  }
}

inline std::string compilePromptString(
    const std::string& text, const std::optional<std::string>& description,
    const std::vector<uint32_t>& promptSpeechIds = {}) {
  validatePromptInputs(text, description);

  std::ostringstream prompt;
  if (!promptSpeechIds.empty()) {
    prompt << tts_tokens::AUDIO_PROMPT_START_TOKEN;
    appendSpeechTokens(prompt, promptSpeechIds);
    prompt << tts_tokens::AUDIO_PROMPT_END_TOKEN;
  }

  if (description.has_value()) {
    const std::string trimmedDescription = trim(*description);
    prompt << tts_tokens::VOICE_PROMPT_START_TOKEN << trimmedDescription
           << tts_tokens::VOICE_PROMPT_END_TOKEN;
  }

  prompt << tts_tokens::BOT_TOKEN << trim(text)
         << tts_tokens::SPEECH_START_TOKEN;
  return prompt.str();
}

inline std::vector<uint32_t> compilePromptTokens(
    const tt::utils::tokenizers::Tokenizer& tokenizer, const std::string& text,
    const std::optional<std::string>& description,
    const std::vector<uint32_t>& promptSpeechIds = {}) {
  return tokenizer.encode(
      compilePromptString(text, description, promptSpeechIds));
}

}  // namespace tt::utils::tts_prompt_compiler
