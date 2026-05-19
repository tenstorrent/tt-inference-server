// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "domain/llm/chat_message.hpp"

namespace tt::utils {

using namespace tt::domain::llm;

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
std::vector<ChatMessage> stripToolMessages(
    const std::vector<ChatMessage>& messages);

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
std::optional<std::vector<ChatMessage>> extractPriorTurnPrefix(
    const std::vector<ChatMessage>& messages);

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
uint64_t hashConversationPrefix(const std::vector<ChatMessage>& prefix);

/**
 * Render the LAST user message on its own, with addGenerationPrompt=true.
 *
 * BOS handling: when hasPriorTurn is true, the leading BOS produced by the
 * chat template is stripped because it is already in the slot's KV cache.
 * When false, BOS is kept so the model sees the start-of-sequence marker on
 * the fresh conversation.
 *
 * @param messages Input chat messages (typically ends with user message)
 * @param hasPriorTurn True iff the conversation continues a previously-cached
 *     [assistant, user] turn (see extractPriorTurnPrefix).
 * @return Rendered delta prompt for the last user turn
 */
std::string renderLastUserTurn(const std::vector<ChatMessage>& messages,
                               bool hasPriorTurn);

/**
 * Routing information computed from conversation messages for prefix caching.
 * Used by the controller to determine session lookup and registration.
 *
 * `deltaPrompt` is a variant so the same struct serves both inputs:
 *   - message-path (HTTP /v1/chat/completions, /v1/responses): a rendered
 *     string ready for tokenization.
 *   - tokens-path (Dynamo `generate`): the suffix of pre-tokenized ids that
 *     the worker still needs to prefill on a continuation.
 */
struct PrefixCachingInfo {
  std::optional<uint64_t>
      lookupHash;  // Hash of prior-turn prefix (for session lookup)
  uint64_t registrationHash =
      0;  // Hash of current conversation (for next turn's lookup)
  std::variant<std::string, std::vector<int>>
      deltaPrompt;  // Last user turn rendered (string) or delta token ids
  bool hasPriorTurn =
      false;  // True if a prior assistant turn exists (enables lookup)
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
    const std::vector<ChatMessage>& messages);

// ---------------------------------------------------------------------------
// Token-level helpers (used by the Dynamo backend, where the frontend has
// already applied the chat template and we receive only token ids).
//
// Conceptually these mirror the message-level helpers above:
//   stripToolMessages    -> N/A (tool turns are already templated into tokens)
//   extractPriorTurnPrefix -> extractPriorTurnPrefixTokens
//   hashConversationPrefix -> hashTokenPrefix
//   computePrefixCachingInfo -> computePrefixCachingInfoFromTokens
// ---------------------------------------------------------------------------

/**
 * Stable 64-bit hash over a sequence of token ids. Computed from the byte
 * representation of the int sequence so two callers produce matching hashes
 * iff the underlying token ids are identical.
 */
uint64_t hashTokenPrefix(std::span<const int> tokens);

/**
 * Return the prefix used to LOOK UP a continuing session for a tokenized
 * prompt. The "assistant header sequence" is the multi-token marker that
 * ends every assistant generation prompt in the chat template (e.g.
 * `<|start_header_id|>assistant<|end_header_id|>\n\n` for Llama-3,
 * `<｜Assistant｜>` for DeepSeek). The PROMPT contains:
 *
 *   [...prior turns...] <asst> {a_{n-1}} <user> {u_n} <asst>
 *                       ^^^^^^                       ^^^^^^
 *                       second-to-last               last (current gen prompt)
 *
 * The prior-turn prefix is everything up to and INCLUDING the second-to-last
 * occurrence of the assistant-header sequence. That hash matches what the
 * previous turn registered (its sole `<asst>` becomes the second-to-last
 * here).
 *
 * Returns std::nullopt when:
 *   - assistantHeaderSequence is empty (tokenizer doesn't expose it), or
 *   - the prompt contains fewer than two assistant-header occurrences (no
 *     prior turn).
 */
std::optional<std::vector<int>> extractPriorTurnPrefixTokens(
    std::span<const int> tokens, std::span<const int> assistantHeaderSequence);

/**
 * Compute prefix caching routing information from a tokenized prompt. Mirrors
 * `computePrefixCachingInfo` but operates on token ids: the assistant-header
 * sequence is read from the active tokenizer strategy.
 *
 * @param tokens Full token-id sequence from the request.
 * @return Complete routing information for prefix caching. `deltaPrompt`
 *         holds the suffix tokens (vector<int> variant) on a continuation,
 *         or an empty vector<int> when there is no prior turn.
 */
PrefixCachingInfo computePrefixCachingInfoFromTokens(
    std::span<const int> tokens);

}  // namespace tt::utils
