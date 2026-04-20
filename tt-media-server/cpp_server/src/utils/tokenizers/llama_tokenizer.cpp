// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/llama_tokenizer.hpp"

#include <sstream>

namespace tt::utils::tokenizers {

static const char* llamaHeaderStart = "<|start_header_id|>";
static const char* llamaHeaderEnd = "<|end_header_id|>";
static const char* llamaEot = "<|eot_id|>";
static const char* llamaSystemPreamble =
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n";

std::string LlamaTokenizer::applyChatTemplate(
    const std::vector<tt::domain::ChatMessage>& messages,
    bool addGenerationPrompt,
    const std::optional<std::vector<tt::domain::tool_calls::Tool>>&) const {
  std::ostringstream out;

  std::string systemContent;
  for (const auto& m : messages) {
    if (m.role == "system") {
      if (!systemContent.empty()) systemContent += "\n\n";
      systemContent += m.content;
    }
  }

  if (cfg_.add_bos_token) out << cfg_.bos_token;

  out << llamaHeaderStart << "system" << llamaHeaderEnd << "\n\n"
      << llamaSystemPreamble << systemContent << llamaEot;

  for (const auto& m : messages) {
    if (m.role == "system") continue;
    std::string role = m.role.empty() ? "user" : m.role;
    out << llamaHeaderStart << role << llamaHeaderEnd << "\n\n"
        << m.content << llamaEot;
  }

  if (addGenerationPrompt) {
    out << llamaHeaderStart << "assistant" << llamaHeaderEnd << "\n\n";
  }
  return out.str();
}

}  // namespace tt::utils::tokenizers
