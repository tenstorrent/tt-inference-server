// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "domain/tool.hpp"

namespace tt::services {

/**
 * ToolCallParser for DeepSeek tool calling format.
 *
 * Parses tool calls from model-generated text containing DeepSeek's special
 * markers like <｜tool▁calls▁begin｜>, <｜tool▁call▁begin｜>, etc.
 *
 * This initial implementation provides text-based parsing only (for
 * non-streaming requests). Token-based streaming support can be added later.
 */
class ToolCallParser {
 public:
  // String markers for DeepSeek tool calling format
  static constexpr const char* TOOL_CALLS_BEGIN =
      "<\xEF\xBD\x9C"
      "tool\xE2\x96\x81"
      "calls\xE2\x96\x81"
      "begin\xEF\xBD\x9C>";
  static constexpr const char* TOOL_CALL_BEGIN =
      "<\xEF\xBD\x9C"
      "tool\xE2\x96\x81"
      "call\xE2\x96\x81"
      "begin\xEF\xBD\x9C>";
  static constexpr const char* TOOL_SEP =
      "<\xEF\xBD\x9C"
      "tool\xE2\x96\x81"
      "sep\xEF\xBD\x9C>";
  static constexpr const char* TOOL_CALL_END =
      "<\xEF\xBD\x9C"
      "tool\xE2\x96\x81"
      "call\xE2\x96\x81"
      "end\xEF\xBD\x9C>";
  static constexpr const char* TOOL_CALLS_END =
      "<\xEF\xBD\x9C"
      "tool\xE2\x96\x81"
      "calls\xE2\x96\x81"
      "end\xEF\xBD\x9C>";

  ToolCallParser() = default;
  ~ToolCallParser() = default;

  // Non-copyable (following ReasoningParser pattern)
  ToolCallParser(const ToolCallParser&) = delete;
  ToolCallParser& operator=(const ToolCallParser&) = delete;

  /**
   * Parse complete text to extract tool calls.
   * Used for non-streaming requests.
   *
   * @param text The complete model-generated text
   * @return Optional vector of ToolCall objects, or nullopt if no tool calls
   * found
   */
  std::optional<std::vector<domain::ToolCall>> parseComplete(
      const std::string& text) const;

 private:
  /**
   * Parse a single tool call from extracted text.
   *
   * @param call_text Text for one tool call (between markers)
   * @param index Index for generating tool call ID
   * @return Optional ToolCall object, or nullopt if parsing fails
   */
  std::optional<domain::ToolCall> parseToolCall(const std::string& call_text,
                                                int index) const;

  /**
   * Extract JSON content from markdown code block.
   * Looks for ```json\n{...}\n``` pattern.
   *
   * @param text Text potentially containing markdown code block
   * @return Extracted JSON string, or empty if not found
   */
  std::string extractJsonFromMarkdown(const std::string& text) const;

  /**
   * Trim leading and trailing whitespace.
   *
   * @param text Input string
   * @return Trimmed string
   */
  std::string trimWhitespace(const std::string& text) const;
};

}  // namespace tt::services
