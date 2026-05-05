// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "domain/chat_message.hpp"

namespace tt::utils {

/**
 * Conversation hashing utilities for prefix caching simulation.
 *
 * These functions support content-addressable session routing by computing
 * stable hashes of conversation prefixes and extracting delta prompts.
 */

/**
 * Remove messages whose role is "tool" or "function" (legacy). Order of
 * remaining messages is preserved. Tool-result messages are treated as
 * ephemeral turns that should not participate in prefix identity -- they're
 * deterministic outputs of the preceding assistant's tool_calls, and dropping
 * them lets the [assistant, user] trailing-pair detection work across
 * tool-using conversations.
 *
 * System / developer messages are left intact: they belong to the stable
 * conversation prefix and must contribute to the hash.
 *
 * @param messages Input chat messages
 * @return Messages with tool/function messages filtered out
 */
std::vector<domain::ChatMessage> stripToolMessages(
    const std::vector<domain::ChatMessage>& messages);

/**
 * Return the prefix used to LOOK UP a continuing session: messages with
 * tool/function entries removed and the trailing [assistant, user] pair
 * stripped.
 *
 * Returns std::nullopt when the trailing pair doesn't exist (fresh
 * conversation, or malformed client that sends two user turns in a row).
 * Callers treat nullopt as "skip the lookup, go straight to new session".
 *
 * Precondition: messages.back().role == "user" is expected (new turn).
 * If messages[-2].role != "assistant" after stripping tool/function messages,
 * returns nullopt.
 *
 * @param messages Input chat messages (must end with user message)
 * @return Prior-turn prefix or nullopt if no prior turn exists
 */
std::optional<std::vector<domain::ChatMessage>> extractPriorTurnPrefix(
    const std::vector<domain::ChatMessage>& messages);

/**
 * Stable 64-bit hash of a chat-message prefix, computed from the rendered
 * chat template with addGenerationPrompt=false. Two callers produce matching
 * hashes across turns iff the underlying message list is byte-identical
 * (modulo tool/function turns, which are stripped upstream).
 *
 * @param prefix Chat messages to hash (should already have tool/function
 *               messages stripped)
 * @return 64-bit hash value
 */
uint64_t hashConversationPrefix(const std::vector<domain::ChatMessage>& prefix);

/**
 * Render the LAST user message on its own, with addGenerationPrompt=true and
 * without any BOS/system wrapper. This is the delta sent to the model when
 * the slot's KV cache already contains the prior-turn prefix.
 *
 * @param messages Input chat messages (must end with user message)
 * @return Rendered delta prompt for the last user turn
 */
std::string renderLastUserTurn(
    const std::vector<domain::ChatMessage>& messages);

/**
 * Routing information computed from conversation messages for prefix caching.
 * Used by the controller to determine session lookup and registration.
 */
struct PrefixCachingInfo {
  std::optional<uint64_t>
      lookupHash;  // Hash of prior-turn prefix (for session lookup)
  uint64_t registrationHash =
      0;  // Hash of current conversation (for next turn's lookup)
  std::string deltaPrompt;  // Last user turn rendered (for continuations)
  bool hasPriorTurn =
      false;  // True if assistant messages exist (enables lookup)
};

/**
 * Compute prefix caching routing information from conversation messages.
 * This is the entry point for controllers to extract all routing data needed
 * for hash-based session lookup and registration.
 *
 * @param messages Input chat messages (should end with user message)
 * @return Complete routing information for prefix caching
 */
PrefixCachingInfo computePrefixCachingInfo(
    const std::vector<domain::ChatMessage>& messages);

}  // namespace tt::utils
