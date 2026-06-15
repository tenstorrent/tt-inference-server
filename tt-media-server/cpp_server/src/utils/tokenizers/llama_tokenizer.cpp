// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/llama_tokenizer.hpp"

#include <sstream>
#include <stdexcept>

namespace tt::utils::tokenizers {

static const char* llamaHeaderStart = "<|start_header_id|>";
static const char* llamaHeaderEnd = "<|end_header_id|>";
static const char* llamaEot = "<|eot_id|>";
static const char* llamaSystemPreamble =
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n";

std::string LlamaTokenizer::applyChatTemplate(
    const std::vector<tt::domain::llm::ChatMessage>& messages,
    bool addGenerationPrompt, [[maybe_unused]] bool enableReasoning,
    bool skipApplyChatTemplate) const {
  std::ostringstream out;

  if (skipApplyChatTemplate) {
    for (const auto& m : messages) {
      out << m.content;
    }
    return out.str();
  }

  // Extract system message
  std::string systemContent;
  auto filteredMessages = messages;
  if (!messages.empty() && messages[0].role == "system") {
    systemContent = messages[0].content;
    filteredMessages = std::vector<tt::domain::llm::ChatMessage>(
        messages.begin() + 1, messages.end());
  }

  // BOS token
  if (cfg_.add_bos_token) out << cfg_.bos_token;

  // System header
  out << llamaHeaderStart << "system" << llamaHeaderEnd << "\n\n";

  out << llamaSystemPreamble << systemContent << llamaEot;

  // Process remaining messages
  for (const auto& m : filteredMessages) {
    // Regular user/assistant message
    std::string role = m.role.empty() ? "user" : m.role;
    out << llamaHeaderStart << role << llamaHeaderEnd << "\n\n";
    out << m.content << llamaEot;
  }

  // Add generation prompt
  if (addGenerationPrompt) {
    out << llamaHeaderStart << "assistant" << llamaHeaderEnd << "\n\n";
  }

  return out.str();
}

std::vector<int> LlamaTokenizer::assistantHeaderSequence() const {
  // Llama-3 chat template ends every assistant generation prompt with the
  // multi-token sequence `<|start_header_id|>assistant<|end_header_id|>\n\n`.
  // Encode it once on first use; the result is stable for the process
  // lifetime so it's safe to cache (per Tokenizer instance — Tokenizer is
  // thread-local).
  static thread_local std::vector<int> cached;
  if (cached.empty()) {
    std::string header =
        std::string(llamaHeaderStart) + "assistant" + llamaHeaderEnd + "\n\n";
    cached = encode(header);
  }
  return cached;
}

}  // namespace tt::utils::tokenizers
