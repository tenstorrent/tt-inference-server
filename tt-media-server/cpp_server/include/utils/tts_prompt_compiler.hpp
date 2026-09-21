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

// TTS-2 prompt format (2026-09 / v2):
//   TVD / description-only:
//     <|voice_prompt_start|>{description}<|voice_prompt_end|>
//     <|bot|>[<|instruction_start|>{instruction}<|instruction_end|>]{text}
//     <|reserved_token_0..7|><|speech_start|>
//   Voice-clone continuation:
//     <|audio_prompt_start|><|s_12|><|s_34|>...<|audio_prompt_end|>
//     <|voice_prompt_start|>{description}<|voice_prompt_end|>
//     <|bot|>[instruction]{text}<|reserved_token_0..7|><|speech_start|>
//
// Two things changed in v2 and both are silent if wrong -- the model simply
// produces worse audio, with no error anywhere:
//   * an instruction is WRAPPED in instruction tokens, not rendered as a
//     bracketed "[say with anger] ..." prefix;
//   * eight readout tokens precede every generated speech segment.
// Mirrors the reference at tts-models/prompting.py.
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

// Renders a turn's instruction together with its transcript. Empty
// instruction returns the text unchanged; note the wrapped form concatenates
// with NO separator, matching the reference.
inline std::string formatInstruction(const std::string& text,
                                     const std::string& instruction) {
  const std::string trimmedInstruction = trim(instruction);
  if (trimmedInstruction.empty()) {
    return text;
  }
  const std::string wrapped = std::string(tts_tokens::INSTRUCTION_START_TOKEN) +
                              trimmedInstruction +
                              tts_tokens::INSTRUCTION_END_TOKEN;
  return text.empty() ? wrapped : wrapped + text;
}

inline std::string compilePromptString(
    const std::string& text, const std::optional<std::string>& description,
    const std::vector<uint32_t>& promptSpeechIds = {},
    const std::string& bosToken = "", const std::string& instruction = "") {
  validatePromptInputs(text, description);

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

  if (description.has_value()) {
    const std::string trimmedDescription = trim(*description);
    prompt << tts_tokens::VOICE_PROMPT_START_TOKEN << trimmedDescription
           << tts_tokens::VOICE_PROMPT_END_TOKEN;
  }

  prompt << tts_tokens::BOT_TOKEN << formatInstruction(trim(text), instruction)
         << tts_tokens::readoutTokens() << tts_tokens::SPEECH_START_TOKEN;
  return prompt.str();
}

inline std::vector<uint32_t> compilePromptTokens(
    const tt::utils::tokenizers::Tokenizer& tokenizer, const std::string& text,
    const std::optional<std::string>& description,
    const std::vector<uint32_t>& promptSpeechIds = {},
    const std::string& bosToken = "", const std::string& instruction = "") {
  return tokenizer.encode(compilePromptString(text, description,
                                              promptSpeechIds, bosToken,
                                              instruction));
}

}  // namespace tt::utils::tts_prompt_compiler
