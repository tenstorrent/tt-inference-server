// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/tool_call/mock_tool_call_parser.hpp"

#include <json/json.h>

namespace tt::services::tool_call {

std::optional<std::vector<domain::ToolCall>>
MockToolCallParser::parseComplete(const std::string& text) const {
  size_t calls_begin = text.find(TOOL_CALLS_BEGIN);
  if (calls_begin == std::string::npos) {
    return std::nullopt;
  }

  size_t calls_end = text.find(TOOL_CALLS_END, calls_begin);
  if (calls_end == std::string::npos) {
    return std::nullopt;
  }

  size_t content_start = calls_begin + std::string(TOOL_CALLS_BEGIN).length();
  std::string calls_block =
      text.substr(content_start, calls_end - content_start);

  std::vector<domain::ToolCall> tool_calls;
  size_t pos = 0;
  int index = 0;

  while (true) {
    size_t call_begin = calls_block.find(TOOL_CALL_BEGIN, pos);
    if (call_begin == std::string::npos) {
      break;
    }

    size_t call_content_start =
        call_begin + std::string(TOOL_CALL_BEGIN).length();
    size_t call_end = calls_block.find(TOOL_CALL_END, call_content_start);
    if (call_end == std::string::npos) {
      break;
    }

    std::string call_text =
        calls_block.substr(call_content_start, call_end - call_content_start);

    auto tool_call = parseToolCall(call_text, index);
    if (tool_call.has_value()) {
      tool_calls.push_back(std::move(tool_call.value()));
      index++;
    }

    pos = call_end + std::string(TOOL_CALL_END).length();
  }

  if (tool_calls.empty()) {
    return std::nullopt;
  }

  return tool_calls;
}

std::optional<domain::ToolCall> MockToolCallParser::parseToolCall(
    const std::string& call_text, int index) const {
  size_t sep_pos = call_text.find(TOOL_SEP);
  if (sep_pos == std::string::npos) {
    return std::nullopt;
  }

  std::string type = trimWhitespace(call_text.substr(0, sep_pos));

  std::string rest = call_text.substr(sep_pos + std::string(TOOL_SEP).length());

  size_t newline_pos = rest.find('\n');
  std::string function_name;
  std::string args_section;

  if (newline_pos != std::string::npos) {
    function_name = trimWhitespace(rest.substr(0, newline_pos));
    args_section = rest.substr(newline_pos + 1);
  } else {
    function_name = trimWhitespace(rest);
    args_section = "";
  }

  std::string arguments = extractJsonFromMarkdown(args_section);

  if (!arguments.empty()) {
    Json::Reader reader;
    Json::Value test;
    if (!reader.parse(arguments, test)) {
      arguments = "{}";
    }
  } else {
    arguments = "{}";
  }

  domain::ToolCall tool_call;
  tool_call.id = "call_" + std::to_string(index);
  tool_call.type = type;
  tool_call.function.name = function_name;
  tool_call.function.arguments = arguments;

  return tool_call;
}

std::string MockToolCallParser::extractJsonFromMarkdown(
    const std::string& text) const {
  size_t json_start = text.find("```json");
  if (json_start == std::string::npos) {
    json_start = text.find("```");
    if (json_start == std::string::npos) {
      return "";
    }
    json_start += 3;
  } else {
    json_start += 7;
  }

  while (json_start < text.length() &&
         (text[json_start] == '\n' || text[json_start] == '\r' ||
          text[json_start] == ' ' || text[json_start] == '\t')) {
    json_start++;
  }

  size_t json_end = text.find("```", json_start);
  if (json_end == std::string::npos) {
    json_end = text.length();
  }
  std::string json = text.substr(json_start, json_end - json_start);
  return trimWhitespace(json);
}

std::string MockToolCallParser::trimWhitespace(
    const std::string& text) const {
  if (text.empty()) return text;

  size_t first = text.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) {
    return "";
  }

  size_t last = text.find_last_not_of(" \t\n\r");
  return text.substr(first, last - first + 1);
}

std::string MockToolCallParser::stripMarkers(const std::string& text) const {
  size_t calls_begin = text.find(TOOL_CALLS_BEGIN);
  if (calls_begin == std::string::npos) {
    return text;
  }

  size_t calls_end = text.find(TOOL_CALLS_END, calls_begin);
  if (calls_end == std::string::npos) {
    return text;
  }

  calls_end += std::string(TOOL_CALLS_END).length();
  std::string before = text.substr(0, calls_begin);
  std::string after =
      calls_end < text.length() ? text.substr(calls_end) : "";
  return before + after;
}

}  // namespace tt::services::tool_call
