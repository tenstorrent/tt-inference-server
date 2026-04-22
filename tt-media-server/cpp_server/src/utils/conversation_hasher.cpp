// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/conversation_hasher.hpp"

#include <functional>

#include "utils/tokenizers/tokenizer.hpp"

namespace tt::utils {

std::vector<domain::ChatMessage> stripToolMessages(
    const std::vector<domain::ChatMessage>& messages) {
  std::vector<domain::ChatMessage> result;
  result.reserve(messages.size());

  for (const auto& msg : messages) {
    if (msg.role != "tool" && msg.role != "function") {
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

  // Drop tool/function turns so the trailing-pair detection works across
  // tool-using conversations. System/developer messages are kept because they
  // belong to the stable prefix identity.
  auto turns = stripToolMessages(messages);

  // Need at least 2 messages: [assistant, user] to strip
  if (turns.size() < 2) {
    return std::nullopt;
  }

  // Check if second-to-last is assistant
  if (turns[turns.size() - 2].role != "assistant") {
    return std::nullopt;
  }

  // Remove the trailing [assistant, user] pair
  std::vector<domain::ChatMessage> priorPrefix;
  priorPrefix.reserve(turns.size() - 2);

  for (size_t i = 0; i < turns.size() - 2; ++i) {
    priorPrefix.push_back(turns[i]);
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
  return std::hash<std::string>()(rendered);
}

std::string renderLastUserTurn(
    const std::vector<domain::ChatMessage>& messages) {
  auto it = std::find_if(
      messages.rbegin(), messages.rend(),
      [](const domain::ChatMessage& msg) { return msg.role == "user"; });
  if (it == messages.rend()) {
    return "";
  }
  const auto& tokenizer = tokenizers::activeTokenizer();
  return tokenizer.applyChatTemplate({*it}, true);
}

PrefixCachingInfo computePrefixCachingInfo(
    const std::vector<domain::ChatMessage>& messages) {
  PrefixCachingInfo info;

  // Drop tool/function turns before hashing; system/developer messages stay
  // as part of the stable prefix identity.
  auto turns = stripToolMessages(messages);

  // deltaPrompt is the last user turn
  info.deltaPrompt = renderLastUserTurn(turns);

  // registrationHash = always hash of full current conversation
  info.registrationHash = hashConversationPrefix(turns);

  // Try to extract prior turn prefix (excluding last [assistant, user] pair)
  auto priorPrefix = extractPriorTurnPrefix(messages);
  if (priorPrefix.has_value()) {
    info.hasPriorTurn = true;
    info.lookupHash = hashConversationPrefix(*priorPrefix);
  } else {
    info.hasPriorTurn = false;
  }

  return info;
}

}  // namespace tt::utils
