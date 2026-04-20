// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/conversation_hasher.hpp"

#include <functional>

#include "utils/tokenizers/tokenizer.hpp"

namespace tt::utils {

std::vector<domain::ChatMessage> stripSystemMessages(
    const std::vector<domain::ChatMessage>& messages) {
  std::vector<domain::ChatMessage> result;
  result.reserve(messages.size());

  for (const auto& msg : messages) {
    if (msg.role != "system") {
      result.push_back(msg);
    }
  }

  return result;
}

std::optional<std::vector<domain::ChatMessage>> extractPriorTurnPrefix(
    const std::vector<domain::ChatMessage>& messages) {
  // Precondition check: messages should end with user
  if (messages.empty() || messages.back().role != "user") {
    return std::nullopt;
  }

  // Strip system messages first
  auto nonSystemMessages = stripSystemMessages(messages);

  // Need at least 2 messages: [assistant, user] to strip
  if (nonSystemMessages.size() < 2) {
    return std::nullopt;
  }

  // Check if second-to-last is assistant
  if (nonSystemMessages[nonSystemMessages.size() - 2].role != "assistant") {
    return std::nullopt;
  }

  // Remove the trailing [assistant, user] pair
  std::vector<domain::ChatMessage> priorPrefix;
  priorPrefix.reserve(nonSystemMessages.size() - 2);

  for (size_t i = 0; i < nonSystemMessages.size() - 2; ++i) {
    priorPrefix.push_back(nonSystemMessages[i]);
  }

  // If the result is empty, return nullopt (no prior turn)
  if (priorPrefix.empty()) {
    return std::nullopt;
  }

  return priorPrefix;
}

uint64_t hashConversationPrefix(
    const std::vector<domain::ChatMessage>& prefix) {
  // Empty prefix should have a deterministic hash
  if (prefix.empty()) {
    return 0;
  }

  // Render the prefix with addGenerationPrompt=false
  const auto& tokenizer = tokenizers::activeTokenizer();
  std::string rendered = tokenizer.applyChatTemplate(prefix, false);

  // Compute stable 64-bit hash
  std::hash<std::string> hasher;
  return hasher(rendered);
}

std::string renderLastUserTurn(
    const std::vector<domain::ChatMessage>& messages) {
  // Find the last user message
  for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
    if (it->role == "user") {
      // Render just this message with addGenerationPrompt=true
      std::vector<domain::ChatMessage> singleMessage = {*it};
      const auto& tokenizer = tokenizers::activeTokenizer();
      return tokenizer.applyChatTemplate(singleMessage, true);
    }
  }

  // Should not happen if preconditions are met, but return empty as fallback
  return "";
}

}  // namespace tt::utils
