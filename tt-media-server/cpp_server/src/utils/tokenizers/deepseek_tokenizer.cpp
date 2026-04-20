// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/deepseek_tokenizer.hpp"

#include <sstream>
#include <string>

namespace tt::utils::tokenizers {

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
static const char* dsToolOutputsBegin =
    "<\xEF\xBD\x9C"
    "tool\xE2\x96\x81"
    "outputs\xE2\x96\x81"
    "begin\xEF\xBD\x9C>";
static const char* dsToolOutputBegin =
    "<\xEF\xBD\x9C"
    "tool\xE2\x96\x81"
    "output\xE2\x96\x81"
    "begin\xEF\xBD\x9C>";
static const char* dsToolOutputEnd =
    "<\xEF\xBD\x9C"
    "tool\xE2\x96\x81"
    "output\xE2\x96\x81"
    "end\xEF\xBD\x9C>";
static const char* dsToolOutputsEnd =
    "<\xEF\xBD\x9C"
    "tool\xE2\x96\x81"
    "outputs\xE2\x96\x81"
    "end\xEF\xBD\x9C>";
static const char* dsEndOfSentence =
    "<\xEF\xBD\x9C"
    "end\xE2\x96\x81"
    "of\xE2\x96\x81"
    "sentence\xEF\xBD\x9C>";

std::string DeepseekTokenizer::applyChatTemplate(
    const std::vector<tt::domain::ChatMessage>& messages,
    bool addGenerationPrompt,
    const std::optional<std::vector<tt::domain::tool_calls::Tool>>& tools)
    const {
  std::ostringstream out;

  if (cfg_.add_bos_token) out << cfg_.bos_token;

  for (const auto& m : messages) {
    if (m.role == "system") out << m.content;
  }

  if (tools.has_value() && !tools->empty()) {
    const std::string toolsDescription =
        std::string(
            "You are a helpful assistant with tool calling capabilities. "
            "When a tool call is needed, you MUST use the following format to "
            "issue the call:\n") +
        dsToolCallsBegin + dsToolCallBegin + "function" + dsToolSep +
        "FUNCTION_NAME\n" +
        "```json\n{\"param1\":\"value1\",\"param2\":\"value2\"}\n```" +
        dsToolCallEnd + dsToolCallsEnd +
        "\n\n"
        "Make sure the JSON is valid.\n"
        "## Tools\n\n### Function\n\nYou have the following functions "
        "available:\n\n";

    out << toolsDescription;
    for (const auto& tool : *tools) {
      out << "- `" << tool.functionDefinition.name << "`:\n```json\n"
          << (tool.toJson()) << "\n```\n";
    }
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

}  // namespace tt::utils::tokenizers
