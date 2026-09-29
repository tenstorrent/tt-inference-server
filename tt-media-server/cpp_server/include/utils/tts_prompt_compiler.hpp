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

// TTS-2 prompt format, mirroring the reference compiler
// (tts-models/prompting.py::compile_continuation, single turn, no history):
//   [BOS]
//   <|audio_prompt_start|><|s_12|><|s_34|>...<|audio_prompt_end|>   (if speech IDs)
//   <|voice_prompt_start|>{lowercased description}<|voice_prompt_end|> (if given)
//   <|bot|>{text}
//   <|reserved_token_0|>...<|reserved_token_7|>                       (readout)
//   <|speech_start|>
//
// The final string is tokenized by the TTS tokenizer; speech IDs are encoded as
// literal tokenizer tokens like <|s_123|>, not inserted as raw token IDs. The
// speech IDs are the codec tokens of the voice prompt (50 per second of 16 kHz
// audio) — the output of the reference audio encoder — supplied by the client.
inline std::string trim(std::string value) {
  auto isNotSpace = [](unsigned char ch) { return std::isspace(ch) == 0; };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), isNotSpace));
  value.erase(std::find_if(value.rbegin(), value.rend(), isNotSpace).base(),
              value.end());
  return value;
}

inline std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  return value;
}

inline void appendSpeechTokens(std::ostringstream& prompt,
                               const std::vector<uint32_t>& speechIds) {
  for (uint32_t speechId : speechIds) {
    prompt << tts_tokens::speechTokenForId(speechId);
  }
}

inline void validateSpeechIds(const std::vector<uint32_t>& speechIds) {
  for (uint32_t speechId : speechIds) {
    if (speechId >= tts_tokens::SPEECH_TOKEN_COUNT) {
      throw std::invalid_argument(
          "TTS speech id " + std::to_string(speechId) +
          " is out of range [0, " +
          std::to_string(tts_tokens::SPEECH_TOKEN_COUNT) + ")");
    }
  }
}

inline void validatePromptInputs(
    const std::string& text, const std::optional<std::string>& description,
    const std::vector<uint32_t>& promptSpeechIds = {}) {
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
  validateSpeechIds(promptSpeechIds);
}

inline std::string compilePromptString(
    const std::string& text, const std::optional<std::string>& description,
    const std::vector<uint32_t>& promptSpeechIds = {},
    const std::string& bosToken = "") {
  validatePromptInputs(text, description, promptSpeechIds);

  std::ostringstream prompt;
  // Tokenizer::encode() calls tokenizers-cpp Encode(), which does NOT add
  // special tokens, so BOS has to be part of the prompt string. Without it the
  // model is decoded from a prefix it never saw in training — the reference
  // compiler emits [BOS, <|bot|>, ...] and this produced [<|bot|>, ...].
  if (!bosToken.empty()) {
    prompt << bosToken;
  }
  if (!promptSpeechIds.empty()) {
    prompt << tts_tokens::AUDIO_PROMPT_START_TOKEN;
    appendSpeechTokens(prompt, promptSpeechIds);
    prompt << tts_tokens::AUDIO_PROMPT_END_TOKEN;
  }

  // The reference lowercases the voice prompt: the model was trained on
  // lowercased descriptions.
  if (description.has_value()) {
    prompt << tts_tokens::VOICE_PROMPT_START_TOKEN
           << toLower(trim(*description)) << tts_tokens::VOICE_PROMPT_END_TOKEN;
  }

  // Every generated turn carries the readout prefix ahead of <|speech_start|>,
  // as in training; without it the model is decoded from an unseen prefix.
  prompt << tts_tokens::BOT_TOKEN << trim(text) << tts_tokens::readoutTokens()
         << tts_tokens::SPEECH_START_TOKEN;
  return prompt.str();
}

inline std::vector<uint32_t> compilePromptTokens(
    const tt::utils::tokenizers::Tokenizer& tokenizer, const std::string& text,
    const std::optional<std::string>& description,
    const std::vector<uint32_t>& promptSpeechIds = {},
    const std::string& bosToken = "") {
  return tokenizer.encode(
      compilePromptString(text, description, promptSpeechIds, bosToken));
}

}  // namespace tt::utils::tts_prompt_compiler
