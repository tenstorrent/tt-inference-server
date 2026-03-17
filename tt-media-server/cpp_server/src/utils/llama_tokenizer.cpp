// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/llama_tokenizer.hpp"

#include <sstream>

namespace tt::utils {

namespace {
const char* kLlamaHeaderStart = "<|start_header_id|>";
const char* kLlamaHeaderEnd = "<|end_header_id|>";
const char* kLlamaEot = "<|eot_id|>";
const char* kLlamaSystemPreamble =
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n";
}  // namespace

std::string LlamaTokenizer::applyChatTemplate(
    const std::vector<domain::ChatMessage>& messages,
    bool addGenerationPrompt) const {
  std::ostringstream out;

  std::string systemContent;
  for (const auto& m : messages) {
    if (m.role == "system") {
      if (!systemContent.empty()) systemContent += "\n\n";
      systemContent += m.content;
    }
  }

  if (cfg.add_bos_token) out << cfg.bos_token;

  out << kLlamaHeaderStart << "system" << kLlamaHeaderEnd << "\n\n"
      << kLlamaSystemPreamble << systemContent << kLlamaEot;

  for (const auto& m : messages) {
    if (m.role == "system") continue;
    std::string role = m.role.empty() ? "user" : m.role;
    out << kLlamaHeaderStart << role << kLlamaHeaderEnd << "\n\n"
        << m.content << kLlamaEot;
  }

  if (addGenerationPrompt) {
    out << kLlamaHeaderStart << "assistant" << kLlamaHeaderEnd << "\n\n";
  }
  return out.str();
}

}  // namespace tt::utils
