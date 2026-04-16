// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "services/tool_call/tool_call_parser.hpp"

namespace tt::services::tool_call {

/**
 * DeepSeek-specific tool call parser.
 * Parses DeepSeek's format with special markers like <｜tool▁calls▁begin｜>
 */
class DeepSeekToolCallParser : public ToolCallParser {
 public:
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

  DeepSeekToolCallParser() = default;
  ~DeepSeekToolCallParser() override = default;

  std::optional<std::vector<domain::ToolCall>> parseComplete(
      const std::string& text) const override;

  std::string stripMarkers(const std::string& text) const override;

 private:
  std::optional<domain::ToolCall> parseToolCall(const std::string& call_text,
                                                int index) const;
  std::string extractJsonFromMarkdown(const std::string& text) const;
  std::string trimWhitespace(const std::string& text) const;
};

}  // namespace tt::services::tool_call
