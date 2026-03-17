// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/deepseek_tokenizer.hpp"

#include <sstream>

namespace tt::utils {

namespace {
const char* kDsUserTag =
    "<\xEF\xBD\x9C"
    "User\xEF\xBD\x9C>";
const char* kDsAssistantTag =
    "<\xEF\xBD\x9C"
    "Assistant\xEF\xBD\x9C>";
}  // namespace

std::string DeepseekTokenizer::applyChatTemplate(
    const std::vector<domain::ChatMessage>& messages,
    bool addGenerationPrompt) const {
  std::ostringstream out;

  if (cfg.add_bos_token) out << cfg.bos_token;

  for (const auto& m : messages) {
    if (m.role == "system") out << m.content;
  }

  for (const auto& m : messages) {
    if (m.role == "system") continue;
    if (m.role == "user") {
      out << kDsUserTag << m.content;
    } else if (m.role == "assistant") {
      out << kDsAssistantTag << m.content;
      if (cfg.add_eos_token) out << cfg.eos_token;
    }
  }

  if (addGenerationPrompt) {
    out << kDsAssistantTag;
  }
  return out.str();
}

}  // namespace tt::utils
