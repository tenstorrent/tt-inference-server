// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config/types.hpp"

namespace tt::services {

/**
 * Interface for parsing model-specific tool call formats from generated text.
 * Each model (DeepSeek, Llama, etc.) has its own format for tool calls.
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
   * Build a forced tool call (tool_choice="function") from model output text.
   * Parses the text as JSON to extract arguments if possible, then wraps the
   * result in OpenAI tool_calls format.
   *
   * Returns a Json::Value array with a single tool call entry.
   */
  virtual Json::Value buildForcedToolCall(
      const std::string& text, const std::string& functionName) const;
};

/**
 * Factory function to create the appropriate parser based on model type.
 */
std::unique_ptr<IToolCallParser> createToolCallParser(
    tt::config::ModelType modelType);

}  // namespace tt::services
