// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/qwen_tokenizer.hpp"

#include <sstream>

namespace tt::utils::tokenizers {

std::string QwenTokenizer::applyChatTemplate(
    const std::vector<tt::domain::llm::ChatMessage>& messages,
    bool addGenerationPrompt,
    const std::optional<std::vector<tt::domain::tool_calls::Tool>>& /*tools*/,
    [[maybe_unused]] bool enableReasoning, bool skipApplyChatTemplate) const {
  if (skipApplyChatTemplate) {
    std::ostringstream out;
    for (const auto& m : messages) {
      out << m.content;
    }
    return out.str();
  }

  // Use the tokenizer's built-in chat template (from tokenizer_config.json)
  // The base Tokenizer class handles this via the jinja template
  std::ostringstream out;
  for (size_t i = 0; i < messages.size(); ++i) {
    const auto& m = messages[i];
    if (m.role == "system") {
      out << "<|im_start|>system\n" << m.content << "<|im_end|>\n";
    } else if (m.role == "user") {
      out << "<|im_start|>user\n" << m.content << "<|im_end|>\n";
    } else if (m.role == "assistant") {
      out << "<|im_start|>assistant\n" << m.content << "<|im_end|>\n";
    }
  }
  if (addGenerationPrompt) {
    out << "<|im_start|>assistant\n";
  }
  return out.str();
}

}  // namespace tt::utils::tokenizers
