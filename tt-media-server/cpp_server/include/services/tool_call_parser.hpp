// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config/types.hpp"

namespace tt::services {

// State for structured output streaming - filters out {"arguments": wrapper
enum class StructuredOutputState {
  SKIPPING_PREFIX,  // Accumulating and matching {"arguments":
  STREAMING,        // Streaming the actual arguments content
  DONE              // Finished (skipping trailing wrapper)
};

struct StructuredOutputParseState {
  StructuredOutputState state = StructuredOutputState::SKIPPING_PREFIX;
  std::string buffer;        // Accumulation buffer for prefix matching
  int brace_depth = 0;       // Track nested braces to know when arguments end
  bool sent_start = false;   // Whether TOOL_CALL_START has been sent
  std::string tool_call_id;  // Generated tool call ID
};

// Type of content being generated
enum class ToolCallContentType {
  TOOL_CALL,  // Inside tool call block
  REGULAR     // Outside tool call block (normal content)
};

// Type of tool call delta for streaming
enum class ToolCallDeltaType {
  NONE,             // Not in a tool call
  TOOL_CALL_START,  // Starting a new tool call (send structure with id, type,
                    // function.name)
  ARGUMENTS_DELTA,  // Adding to function.arguments
  TOOL_CALL_END     // Ending current tool call
};

// Result of processing a single token for tool calls
struct ToolCallTokenResult {
  ToolCallContentType type;  // Type of content (tool call or regular)
  std::string text;          // Decoded text for this token (or arguments delta)
  bool should_emit;          // Whether to send to client
  ToolCallDeltaType delta_type;  // Type of streaming delta to send
  int tool_call_index;           // Index of current tool call (0-based)
  std::string function_name;     // Function name (for TOOL_CALL_START)
  std::string tool_call_id;      // Tool call ID (for TOOL_CALL_START)
  std::optional<Json::Value>
      tool_calls_delta;  // Pre-built tool_calls array for the choice
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
   * Parse tool calls from complete generated text.
   * Returns an array of tool call objects in OpenAI format, or std::nullopt if
   * no tool calls found.
   *
   * Expected output format:
   * [
   *   {
   *     "id": "call_0",
   *     "type": "function",
   *     "function": {
   *       "name": "get_weather",
   *       "arguments": "{\"location\":\"San Francisco\"}"
   *     }
   *   }
   * ]
   */
  virtual std::optional<Json::Value> parseComplete(
      const std::string& text, bool parallelToolCalls = true) const = 0;

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
  virtual void initializeTask(uint32_t task_id,
                              [[maybe_unused]] const std::string& function_name) {
    // Default implementation ignores function_name
    initializeTask(task_id);
  }

  /**
   * Process single token for streaming.
   * Returns content type and whether to emit.
   *
   * @param task_id Unique task identifier
   * @param token_id Token ID to process
   * @param decoded_text Decoded text for this token
   * @return ToolCallTokenResult with content type and emit flag
   */
  virtual ToolCallTokenResult processToken(uint32_t task_id, int64_t token_id,
                                           const std::string& decoded_text) = 0;

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
 * Factory function to create the appropriate parser based on model type.
 * Used for natural tool calls with model-specific markers (e.g., DeepSeek).
 */
std::unique_ptr<IToolCallParser> createToolCallParser(
    tt::config::ModelType modelType);

/**
 * Create a JSON tool call parser for structured output.
 * Handles models outputting {"arguments": {...}} wrapper format.
 * Used when tool_choice is "function" or "required".
 */
std::unique_ptr<IToolCallParser> createJsonToolCallParser();

}  // namespace tt::services
