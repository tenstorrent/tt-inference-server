// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/conversation_hasher.hpp"

#include <algorithm>

#define XXH_INLINE_ALL
#include "utils/tokenizers/tokenizer.hpp"
#include "xxhash.h"

namespace tt::utils {

std::vector<ChatMessage> stripToolMessages(
    const std::vector<ChatMessage>& messages) {
  std::vector<ChatMessage> result;
  result.reserve(messages.size());

  for (const auto& msg : messages) {
    if (msg.role != "tool" && msg.role != "function") {
      result.push_back(msg);
    }
  }

  return result;
}

std::optional<std::vector<ChatMessage>> extractPriorTurnPrefix(
    const std::vector<ChatMessage>& messages) {
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
  std::vector<ChatMessage> priorPrefix;
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

uint64_t hashConversationPrefix(const std::vector<ChatMessage>& prefix) {
  // Empty prefix should have a deterministic hash
  if (prefix.empty()) {
    return 0;
  }

  // Render the prefix with addGenerationPrompt=false
  const auto& tokenizer = tokenizers::activeTokenizer();
  std::string rendered = tokenizer.applyChatTemplate(prefix, false);

  // Compute stable 64-bit hash using xxHash64 (deterministic across
  // platforms/restarts)
  return XXH64(rendered.data(), rendered.size(), 0);
}

std::string renderLastUserTurn(const std::vector<ChatMessage>& messages,
                               bool hasPriorTurn) {
  auto it =
      std::find_if(messages.rbegin(), messages.rend(),
                   [](const ChatMessage& msg) { return msg.role == "user"; });
  if (it == messages.rend()) {
    return "";
  }
  const auto& tokenizer = tokenizers::activeTokenizer();
  std::string rendered = tokenizer.applyChatTemplate({*it}, true);

  // applyChatTemplate prepends BOS based on the tokenizer config. For
  // continuations BOS is already in the slot's KV cache and must not be
  // duplicated in the delta; for fresh conversations keep it so the model
  // sees the start-of-sequence marker. The BOS string is fixed for the
  // process lifetime, so cache it to avoid copying TokenizerConfig on every
  // call.
  static const std::string bosToken = tokenizers::getTokenizerConfig().bos_token;
  if (hasPriorTurn && !bosToken.empty() &&
      rendered.compare(0, bosToken.size(), bosToken) == 0) {
    rendered.erase(0, bosToken.size());
  }
  return rendered;
}

PrefixCachingInfo computePrefixCachingInfo(
    const std::vector<ChatMessage>& messages) {
  PrefixCachingInfo info;

  // Drop tool/function turns before hashing; system/developer messages stay
  // as part of the stable prefix identity.
  auto turns = stripToolMessages(messages);

  // Determine prior-turn status first; the renderer needs it to decide
  // whether to keep the BOS token in the delta prompt.
  auto priorPrefix = extractPriorTurnPrefix(messages);
  info.hasPriorTurn = priorPrefix.has_value();
  if (info.hasPriorTurn) {
    info.lookupHash = hashConversationPrefix(*priorPrefix);
  }

  info.deltaPrompt = renderLastUserTurn(turns, info.hasPriorTurn);
  info.registrationHash = hashConversationPrefix(turns);

  return info;
}

}  // namespace tt::utils
