// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "domain/llm/chat_message.hpp"

namespace tt::api::resolvers {

/**
 * Prefix-cache routing helpers used by ChatCompletionsResolver. The
 * functions are exposed so resolver tests can prime the SessionManager
 * with deterministic hashes; production code should call
 * `computePrefixCachingInfo()` rather than the lower-level helpers
 * directly.
 */

/**
 * Remove messages whose role is "tool" or "function" (legacy). Order of
 * remaining messages is preserved. Tool-result messages are treated as
 * ephemeral turns that should not participate in prefix identity --
 * they're deterministic outputs of the preceding assistant's
 * tool_calls, and dropping them lets the [assistant, user] trailing-pair
 * detection work across tool-using conversations.
 *
 * System / developer messages are left intact: they belong to the
 * stable conversation prefix and must contribute to the hash.
 */
std::vector<domain::llm::ChatMessage> stripToolMessages(
    const std::vector<domain::llm::ChatMessage>& messages);

/**
 * Return the prefix used to LOOK UP a continuing session: messages with
 * tool/function entries removed and the trailing [assistant, user] pair
 * stripped.
 *
 * Returns nullopt when the trailing pair doesn't exist (fresh
 * conversation, or malformed client that sends two user turns in a
 * row). Callers treat nullopt as "skip the lookup, go straight to new
 * session".
 *
 * Precondition: messages.back().role == "user" is expected (new turn).
 * If messages[-2].role != "assistant" after stripping tool/function
 * messages, returns nullopt.
 */
std::optional<std::vector<domain::llm::ChatMessage>> extractPriorTurnPrefix(
    const std::vector<domain::llm::ChatMessage>& messages);

/**
 * Stable 64-bit hash of a chat-message prefix, computed from the
 * rendered chat template with addGenerationPrompt=false. Two callers
 * produce matching hashes across turns iff the underlying message list
 * is byte-identical (modulo tool/function turns, which are stripped
 * upstream).
 */
uint64_t hashConversationPrefix(
    const std::vector<domain::llm::ChatMessage>& prefix);

/**
 * Render the LAST user message on its own, with addGenerationPrompt=true
 * and without any BOS/system wrapper. This is the delta sent to the
 * model when the slot's KV cache already contains the prior-turn
 * prefix.
 */
std::string renderLastUserTurn(
    const std::vector<domain::llm::ChatMessage>& messages);

/**
 * Routing information derived from conversation messages. Used by the
 * resolver to drive prefix-cache lookup and registration.
 */
struct PrefixCachingInfo {
  std::optional<uint64_t> lookupHash;  // Hash of prior-turn prefix (lookup key)
  uint64_t registrationHash =
      0;                      // Hash of current conversation (next-turn key)
  std::string deltaPrompt;    // Last user turn rendered (delta)
  bool hasPriorTurn = false;  // True iff an [assistant, user] pair was found
};

PrefixCachingInfo computePrefixCachingInfo(
    const std::vector<domain::llm::ChatMessage>& messages);

}  // namespace tt::api::resolvers
