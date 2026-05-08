// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "config/types.hpp"

namespace tt::services {

// Type of tool call delta for streaming
enum class ToolCallDeltaType {
  TOOL_CALL_START,  // Starting a new tool call (id, type, function.name)
  ARGUMENTS_DELTA,  // Adding to function.arguments
  TOOL_CALL_END     // Ending current tool call
};

// Tool call delta to emit to client
struct ToolCallTokenResult {
  ToolCallDeltaType delta_type;  // Type of streaming delta
  int tool_call_index;           // Index of current tool call (0-based)
  std::string text;              // Arguments delta text
  std::string function_name;     // Function name (for TOOL_CALL_START)
  std::string tool_call_id;      // Tool call ID (for TOOL_CALL_START)
  Json::Value tool_calls_delta;  // Pre-built tool_calls array for the choice
};

/**
 * Interface for parsing model-specific tool call formats from generated text.
 * Each model (DeepSeek, Llama, etc.) has its own format for tool calls.
 *
 * Supports both complete text parsing (non-streaming) and token-by-token
 * streaming parsing.
 */
class IToolCallParser {
 public:
  virtual ~IToolCallParser() = default;

  /**
   * Strip tool call markers from the text, leaving only regular content.
   * Used to clean up the message content after extracting tool calls.
   */
  virtual std::string stripMarkers(const std::string& text) const = 0;

  /**
   * Initialize streaming state for a task.
   * Call before processing first token.
   */
  virtual void initializeTask(uint32_t task_id) = 0;

  /**
   * Initialize streaming state for a task with function name context.
   * Used by JsonToolCallParser for structured output where we know
   * the function name upfront from tool_choice.
   */
  virtual void initializeTask(
      uint32_t task_id, [[maybe_unused]] const std::string& function_name) {
    // Default implementation ignores function_name
    initializeTask(task_id);
  }

  /**
   * Process single token for streaming.
   * Returns a tool call delta to emit, or nullopt if nothing to emit.
   *
   * When nullopt is returned, caller should check isInToolCall():
   * - If true: token was consumed by parser, suppress regular output
   * - If false: token is regular content, emit normally
   *
   * @param task_id Unique task identifier
   * @param token_id Token ID to process
   * @param decoded_text Decoded text for this token
   * @return Tool call delta to emit, or nullopt
   */
  virtual std::optional<ToolCallTokenResult> processToken(
      uint32_t task_id, int64_t token_id, const std::string& decoded_text) = 0;

  /**
   * Finalize task state and cleanup.
   * Call when generation completes.
   * Returns the parsed tool calls if any were found, or std::nullopt.
   */
  virtual std::optional<Json::Value> finalizeTask(uint32_t task_id) = 0;

  /**
   * Check if task is currently in tool call mode.
   */
  virtual bool isInToolCall(uint32_t task_id) const = 0;

  /**
   * Get count of active tasks.
   */
  virtual size_t activeTaskCount() const = 0;
};

/**
 * Create a JSON tool call parser for structured output.
 * Handles models outputting {"arguments": {...}} wrapper format.
 * Used when tool_choice is "function" or "required".
 */
std::unique_ptr<IToolCallParser> createJsonToolCallParser();

}  // namespace tt::services
