// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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

    for (const auto& tool : *tools) {
      out << "- `" << tool.functionDefinition.name << "`:\n```json\n"
          << tool.toJson() << "\n```\n";
    }
  }

  bool inToolOutputs = false;
  bool isFirstToolOutput = true;

  for (const auto& m : messages) {
    if (m.role == "system") continue;
    if (m.role == "user") {
      if (inToolOutputs) {
        out << dsToolOutputsEnd;
        inToolOutputs = false;
      }
      out << dsUserTag << m.content;
    } else if (m.role == "assistant") {
      if (inToolOutputs) {
        out << dsToolOutputsEnd;
        inToolOutputs = false;
      }

      // Check if this assistant message has tool calls
      if (m.tool_calls.has_value() && !m.tool_calls->empty()) {
        if (!m.content.empty()) {
          out << dsAssistantTag << m.content;
        }

        out << dsToolCallsBegin;
        for (const auto& toolCall : m.tool_calls.value()) {
          out << dsToolCallBegin << "function" << dsToolSep
              << toolCall.functionCall.name << "\n```json\n"
              << toolCall.functionCall.arguments << "\n```" << dsToolCallEnd;
        }
        out << dsToolCallsEnd << dsEndOfSentence;

      } else {
        out << dsAssistantTag << m.content;
      }
    } else if (m.role == "tool") {
      if (!inToolOutputs) {
        out << dsToolOutputsBegin;
        inToolOutputs = true;
        isFirstToolOutput = true;
      }
      if (!isFirstToolOutput) {
        out << "\n";
      }
      out << dsToolOutputBegin << m.content << dsToolOutputEnd;
      isFirstToolOutput = false;
    }
  }

  if (inToolOutputs) {
    out << dsToolOutputsEnd;
  }
  if (addGenerationPrompt) {
    out << dsAssistantTag;
  }

  return out.str();
}

}  // namespace tt::utils::tokenizers
