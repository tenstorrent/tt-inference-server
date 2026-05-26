// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
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
 */
struct PrefixCachingInfo {
  std::vector<uint64_t> hashes;  // Per-block prefix cache hashes
};

// ---------------------------------------------------------------------------
// Token-level helpers (used by the Dynamo backend, where the frontend has
// already applied the chat template and we receive only token ids).
// ---------------------------------------------------------------------------

/**
 * Stable 64-bit hash over a sequence of token ids. Computed from the byte
 * representation of the int sequence so two callers produce matching hashes
 * iff the underlying token ids are identical.
 */
uint64_t hashTokenPrefix(std::span<const int> tokens);

/**
 * Compute prefix caching routing information from a tokenized prompt.
 * The assistant-header sequence is read from the active tokenizer strategy.
 *
 * @param tokens Full token-id sequence from the request.
 * @return Complete routing information for prefix caching (per-block hashes).
 */
PrefixCachingInfo computePrefixCachingInfoFromTokens(
    std::span<const int> tokens);

/**
 * Compute per-block KV cache hashes using vLLM's prefix caching approach.
 *
 * Tokens are divided into blocks of `kvCacheBlockSize` (from config). Each
 * block's hash is computed as `xxh64(block_tokens, seed=parent_hash)`, where
 * `parent_hash` is the hash of the previous block (0 for the first block).
 * This chaining ensures that two sequences sharing a common prefix produce
 * identical hashes for the shared blocks.
 *
 * Only FULL blocks are hashed — any trailing partial block is ignored (it
 * hasn't been committed to the KV cache yet).
 *
 * @param tokens Full token-id sequence.
 * @return Vector of per-block hashes (one per full block). Empty if the
 *         sequence is shorter than one block.
 */
std::vector<uint64_t> getPrefixCacheHashesByBlocks(std::span<const int> tokens);

}  // namespace tt::utils
