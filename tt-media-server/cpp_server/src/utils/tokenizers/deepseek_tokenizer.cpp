// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/tokenizers/deepseek_tokenizer.hpp"

#include <sstream>

namespace tt::utils::tokenizers {

static const char* dsUserTag =
    "<\xEF\xBD\x9C"
    "User\xEF\xBD\x9C>";
static const char* dsAssistantTag =
    "<\xEF\xBD\x9C"
    "Assistant\xEF\xBD\x9C>";

std::string DeepseekTokenizer::applyChatTemplate(
    const std::vector<tt::domain::ChatMessage>& messages,
    bool addGenerationPrompt) const {
  std::ostringstream out;

  if (cfg_.add_bos_token) out << cfg_.bos_token;

  for (const auto& m : messages) {
    if (m.role == "system") out << m.content;
  }

  for (const auto& m : messages) {
    if (m.role == "system") continue;
    if (m.role == "user") {
      out << dsUserTag << m.content;
    } else if (m.role == "assistant") {
      out << dsAssistantTag << m.content;
      if (cfg_.add_eos_token) out << cfg_.eos_token;
    }
  }

  if (addGenerationPrompt) {
    out << dsAssistantTag;
  }
  return out.str();
}

}  // namespace tt::utils::tokenizers
