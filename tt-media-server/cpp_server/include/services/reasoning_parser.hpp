// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace tt::services {

// Type of content being generated
enum class ContentType {
  REASONING,  // Inside <think>...</think> block
  ANSWER      // Outside reasoning block (normal content)
};

// Result of parsing complete text for reasoning blocks
struct ReasoningParseResult {
  std::optional<std::string> reasoning;  // Reasoning content (inside <think>)
  std::string answer;                    // Answer content (outside <think>)
  bool has_reasoning;                    // Whether reasoning was found
  bool
      is_malformed;  // Whether reasoning block is incomplete (missing </think>)
};

// Result of processing a single token
struct TokenParseResult {
  ContentType type;  // Type of content (reasoning or answer)
  std::string text;  // Decoded text for this token
  bool should_emit;  // Whether to send to client (always true)
};

// Per-task state for token-by-token parsing
struct TaskState {
  bool in_reasoning;      // Currently inside <think> block
  bool seen_think_start;  // Has seen <think> token
};

/**
 * Reasoning Parser for DeepSeek R1 style <think>...</think> tags.
 *
 * This parser tags tokens as REASONING or ANSWER content but sends ALL tokens
 * to the client (matching vLLM behavior). The client decides how to display
 * them.
 *
 * Token-based parsing for streaming (O(1) lookups, no string operations).
 * Text-based parsing for complete responses.
 */
class ReasoningParser {
 public:
  // Token IDs for DeepSeek R1 reasoning markers
  static constexpr int64_t THINK_START_TOKEN = 128798;  // <think>
  static constexpr int64_t THINK_END_TOKEN = 128799;    // </think>
  static constexpr int64_t NEWLINE_TOKEN = 201;         // \n

  // String markers for text-based parsing
  static constexpr const char* THINK_START_TAG = "<think>";
  static constexpr const char* THINK_END_TAG = "</think>";

  ReasoningParser() = default;
  ~ReasoningParser() = default;

  // Non-copyable
  ReasoningParser(const ReasoningParser&) = delete;
  ReasoningParser& operator=(const ReasoningParser&) = delete;

  /**
   * Parse complete text to extract reasoning and answer.
   * Used for non-streaming requests.
   */
  ReasoningParseResult parseComplete(const std::string& text) const;

  /**
   * Initialize streaming state for a task.
   * Call before processing first token.
   */
  void initializeTask(uint32_t task_id);

  /**
   * Process single token for streaming.
   * Returns content type and whether to emit.
   *
   * @param task_id Unique task identifier
   * @param token_id Token ID to process
   * @param decoded_text Decoded text for this token
   * @return TokenParseResult with content type and emit flag
   */
  TokenParseResult processToken(uint32_t task_id, int64_t token_id,
                                const std::string& decoded_text);

  /**
   * Finalize task state and cleanup.
   * Call when generation completes.
   */
  void finalizeTask(uint32_t task_id);

  /**
   * Check if task is currently in reasoning mode.
   */
  bool isInReasoning(uint32_t task_id) const;

  /**
   * Get count of active tasks.
   */
  size_t activeTaskCount() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint32_t, TaskState> task_states_;
};

}  // namespace tt::services
