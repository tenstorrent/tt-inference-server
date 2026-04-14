// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/deepseek_tokenizer.hpp"

#include <json/json.h>

#include <sstream>

namespace tt::utils {

// DeepSeek special tokens using fullwidth characters
// \xEF\xBD\x9C = ｜ (fullwidth vertical bar)
// \xE2\x96\x81 = ▁ (lower one eighth block)
static const char* dsUserTag =
    "<\xEF\xBD\x9C"
    "User\xEF\xBD\x9C>";
static const char* dsAssistantTag =
    "<\xEF\xBD\x9C"
    "Assistant\xEF\xBD\x9C>";
static const char* dsToolCallsBegin =
    "<\xEF\xBD\x9C"
    "tool\xE2\x96\x81"
    "calls\xE2\x96\x81"
    "begin\xEF\xBD\x9C>";
static const char* dsToolCallBegin =
    "<\xEF\xBD\x9C"
    "tool\xE2\x96\x81"
    "call\xE2\x96\x81"
    "begin\xEF\xBD\x9C>";
static const char* dsToolSep =
    "<\xEF\xBD\x9C"
    "tool\xE2\x96\x81"
    "sep\xEF\xBD\x9C>";
static const char* dsToolCallEnd =
    "<\xEF\xBD\x9C"
    "tool\xE2\x96\x81"
    "call\xE2\x96\x81"
    "end\xEF\xBD\x9C>";
static const char* dsToolCallsEnd =
    "<\xEF\xBD\x9C"
    "tool\xE2\x96\x81"
    "calls\xE2\x96\x81"
    "end\xEF\xBD\x9C>";

std::string DeepseekTokenizer::applyChatTemplate(
    const std::vector<domain::ChatMessage>& messages,
    bool addGenerationPrompt,
    const std::optional<std::vector<domain::Tool>>& tools) const {
  std::ostringstream out;

  if (cfg_.add_bos_token) out << cfg_.bos_token;

  // Collect system messages
  for (const auto& m : messages) {
    if (m.role == "system") out << m.content;
  }

  // Append tool template to system prompt if tools are provided
  // Following vLLM/SGLang pattern
  if (tools.has_value() && !tools->empty()) {
    out << "\n\n## Tools\nYou have access to the following tools:\n";

    // Add each tool with its name, description, and parameters
    for (const auto& tool : tools.value()) {
      out << "\n### " << tool.function.name << "\n";
      if (tool.function.description.has_value()) {
        out << "Description: " << tool.function.description.value() << "\n";
      }
      out << "\nParameters: ";

      // Serialize parameters as compact JSON
      Json::StreamWriterBuilder writer;
      writer["indentation"] = "";
      writer["emitUTF8"] = true;
      out << Json::writeString(writer, tool.function.parameters) << "\n";
    }

    // Add format instructions with DeepSeek special tokens
    out << "\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n"
        << dsToolCallsBegin << dsToolCallBegin
        << "tool_call_name" << dsToolSep << "tool_call_arguments"
        << dsToolCallEnd << "{{additional_tool_calls}}" << dsToolCallsEnd
        << "\n\n"
        << "Where:\n\n"
        << "- `tool_call_name` must be an exact match to one of the available "
           "tools\n"
        << "- `tool_call_arguments` must be valid JSON that strictly follows "
           "the tool's Parameters Schema\n"
        << "- For multiple tool calls, chain them directly without separators "
           "or spaces\n";
  }

  for (const auto& m : messages) {
    if (m.role == "system") continue;
    if (m.role == "user") {
      out << dsUserTag << m.content;
    } else if (m.role == "assistant") {
      out << dsAssistantTag << m.content;
      if (cfg_.add_eos_token) out << cfg_.eos_token;
    }
  }

  if (addGenerationPrompt) {
    out << dsAssistantTag;
  }
  return out.str();
}

}  // namespace tt::utils
