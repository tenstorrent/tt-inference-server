// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/tool_call_parser.hpp"

#include <json/json.h>

#include <sstream>

namespace tt::services {

std::optional<std::vector<domain::ToolCall>> ToolCallParser::parseComplete(
    const std::string& text) const {
  // Check if text contains tool calls marker
  size_t calls_begin = text.find(TOOL_CALLS_BEGIN);
  if (calls_begin == std::string::npos) {
    return std::nullopt;  // No tool calls found
  }

  // Find end marker
  size_t calls_end = text.find(TOOL_CALLS_END, calls_begin);
  if (calls_end == std::string::npos) {
    return std::nullopt;  // Malformed - missing end marker
  }

  // Extract tool calls block
  size_t content_start = calls_begin + std::string(TOOL_CALLS_BEGIN).length();
  std::string calls_block = text.substr(content_start, calls_end - content_start);

  // Parse individual tool calls
  std::vector<domain::ToolCall> tool_calls;
  size_t pos = 0;
  int index = 0;

  while (true) {
    // Find next tool call
    size_t call_begin = calls_block.find(TOOL_CALL_BEGIN, pos);
    if (call_begin == std::string::npos) {
      break;  // No more tool calls
    }

    // Find end of this tool call
    size_t call_content_start = call_begin + std::string(TOOL_CALL_BEGIN).length();
    size_t call_end = calls_block.find(TOOL_CALL_END, call_content_start);
    if (call_end == std::string::npos) {
      break;  // Malformed - missing end marker
    }

    // Extract this tool call content
    std::string call_text =
        calls_block.substr(call_content_start, call_end - call_content_start);

    // Parse this tool call
    auto tool_call = parseToolCall(call_text, index);
    if (tool_call.has_value()) {
      tool_calls.push_back(std::move(tool_call.value()));
      index++;
    }

    // Move past this tool call
    pos = call_end + std::string(TOOL_CALL_END).length();
  }

  if (tool_calls.empty()) {
    return std::nullopt;
  }

  return tool_calls;
}

std::optional<domain::ToolCall> ToolCallParser::parseToolCall(
    const std::string& call_text, int index) const {
  // Split by separator to get type and rest
  size_t sep_pos = call_text.find(TOOL_SEP);
  if (sep_pos == std::string::npos) {
    return std::nullopt;  // Malformed - missing separator
  }

  // Extract type (usually "function")
  std::string type = trimWhitespace(call_text.substr(0, sep_pos));

  // Rest contains function name and arguments
  std::string rest = call_text.substr(sep_pos + std::string(TOOL_SEP).length());

  // Extract function name (before newline or code block)
  size_t newline_pos = rest.find('\n');
  std::string function_name;
  std::string args_section;

  if (newline_pos != std::string::npos) {
    function_name = trimWhitespace(rest.substr(0, newline_pos));
    args_section = rest.substr(newline_pos + 1);
  } else {
    // No newline - try to extract what we can
    function_name = trimWhitespace(rest);
    args_section = "";
  }

  // Extract arguments from markdown code block
  std::string arguments = extractJsonFromMarkdown(args_section);

  // Validate JSON
  if (!arguments.empty()) {
    Json::Reader reader;
    Json::Value test;
    if (!reader.parse(arguments, test)) {
      // Invalid JSON - use empty object
      arguments = "{}";
    }
  } else {
    arguments = "{}";
  }

  // Create ToolCall object
  domain::ToolCall tool_call;
  tool_call.id = "call_" + std::to_string(index);
  tool_call.type = type;
  tool_call.function.name = function_name;
  tool_call.function.arguments = arguments;

  return tool_call;
}

std::string ToolCallParser::extractJsonFromMarkdown(
    const std::string& text) const {
  // Look for ```json\n{...}\n```
  size_t json_start = text.find("```json");
  if (json_start == std::string::npos) {
    // Try without language specifier
    json_start = text.find("```");
    if (json_start == std::string::npos) {
      return "";
    }
    json_start += 3;  // Length of "```"
  } else {
    json_start += 7;  // Length of "```json"
  }

  // Skip whitespace/newlines after opening
  while (json_start < text.length() &&
         (text[json_start] == '\n' || text[json_start] == '\r' ||
          text[json_start] == ' ' || text[json_start] == '\t')) {
    json_start++;
  }

  // Find closing ```
  size_t json_end = text.find("```", json_start);
  if (json_end == std::string::npos) {
    // No closing marker - take rest of text
    json_end = text.length();
  }

  // Extract JSON content
  std::string json = text.substr(json_start, json_end - json_start);
  return trimWhitespace(json);
}

std::string ToolCallParser::trimWhitespace(const std::string& text) const {
  if (text.empty()) return text;

  size_t first = text.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) {
    return "";  // All whitespace
  }

  size_t last = text.find_last_not_of(" \t\n\r");
  return text.substr(first, last - first + 1);
}

}  // namespace tt::services
