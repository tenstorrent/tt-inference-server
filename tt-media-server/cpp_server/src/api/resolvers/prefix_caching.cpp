// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/resolvers/prefix_caching.hpp"

#include <algorithm>

#define XXH_INLINE_ALL
#include "utils/tokenizers/tokenizer.hpp"
#include "xxhash.h"

namespace tt::api::resolvers {

std::vector<domain::llm::ChatMessage> stripToolMessages(
    const std::vector<domain::llm::ChatMessage>& messages) {
  std::vector<domain::llm::ChatMessage> result;
  result.reserve(messages.size());

  for (const auto& msg : messages) {
    if (msg.role != "tool" && msg.role != "function") {
      result.push_back(msg);
    }
  }

  return result;
}

std::optional<std::vector<domain::llm::ChatMessage>> extractPriorTurnPrefix(
    const std::vector<domain::llm::ChatMessage>& messages) {
  if (messages.empty() || messages.back().role != "user") {
    return std::nullopt;
  }

  auto turns = stripToolMessages(messages);

  if (turns.size() < 2) {
    return std::nullopt;
  }

  if (turns[turns.size() - 2].role != "assistant") {
    return std::nullopt;
  }

  std::vector<domain::llm::ChatMessage> priorPrefix;
  priorPrefix.reserve(turns.size() - 2);

  for (size_t i = 0; i < turns.size() - 2; ++i) {
    priorPrefix.push_back(turns[i]);
  }

  if (priorPrefix.empty()) {
    return std::nullopt;
  }

  return priorPrefix;
}

uint64_t hashConversationPrefix(
    const std::vector<domain::llm::ChatMessage>& prefix) {
  if (prefix.empty()) {
    return 0;
  }

  const auto& tokenizer = tt::utils::tokenizers::activeTokenizer();
  std::string rendered = tokenizer.applyChatTemplate(prefix, false);

  return XXH64(rendered.data(), rendered.size(), 0);
}

std::string renderLastUserTurn(
    const std::vector<domain::llm::ChatMessage>& messages) {
  auto it = std::find_if(
      messages.rbegin(), messages.rend(),
      [](const domain::llm::ChatMessage& msg) { return msg.role == "user"; });
  if (it == messages.rend()) {
    return "";
  }
  const auto& tokenizer = tt::utils::tokenizers::activeTokenizer();
  return tokenizer.applyChatTemplate({*it}, true);
}

PrefixCachingInfo computePrefixCachingInfo(
    const std::vector<domain::llm::ChatMessage>& messages) {
  PrefixCachingInfo info;

  auto turns = stripToolMessages(messages);

  info.deltaPrompt = renderLastUserTurn(turns);
  info.registrationHash = hashConversationPrefix(turns);

  auto priorPrefix = extractPriorTurnPrefix(messages);
  if (priorPrefix.has_value()) {
    info.hasPriorTurn = true;
    info.lookupHash = hashConversationPrefix(*priorPrefix);
  } else {
    info.hasPriorTurn = false;
  }

  return info;
}

}  // namespace tt::api::resolvers
