// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/llama_tokenizer.hpp"

#include <sstream>

namespace tt::utils {

static const char* LLAMA_HEADER_START = "<|start_header_id|>";
static const char* LLAMA_HEADER_END = "<|end_header_id|>";
static const char* LLAMA_EOT = "<|eot_id|>";
static const char* LLAMA_SYSTEM_PREAMBLE =
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n";

std::string LlamaTokenizer::apply_chat_template(
    const std::vector<domain::ChatMessage>& messages,
    bool add_generation_prompt) const {
  std::ostringstream out;

  std::string system_content;
  for (const auto& m : messages) {
    if (m.role == "system") {
      if (!system_content.empty()) system_content += "\n\n";
      system_content += m.content;
    }
  }

  if (cfg_.add_bos_token) out << cfg_.bos_token;

  out << LLAMA_HEADER_START << "system" << LLAMA_HEADER_END << "\n\n"
      << LLAMA_SYSTEM_PREAMBLE << system_content << LLAMA_EOT;

  for (const auto& m : messages) {
    if (m.role == "system") continue;
    std::string role = m.role.empty() ? "user" : m.role;
    out << LLAMA_HEADER_START << role << LLAMA_HEADER_END << "\n\n"
        << m.content << LLAMA_EOT;
  }

  if (add_generation_prompt) {
    out << LLAMA_HEADER_START << "assistant" << LLAMA_HEADER_END << "\n\n";
  }
  return out.str();
}

}  // namespace tt::utils
