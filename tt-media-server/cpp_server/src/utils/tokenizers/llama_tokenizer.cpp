// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/llama_tokenizer.hpp"

#include <json/json.h>

#include <sstream>
#include <stdexcept>

#include "utils/logger.hpp"

namespace tt::utils::tokenizers {

static const char* llamaHeaderStart = "<|start_header_id|>";
static const char* llamaHeaderEnd = "<|end_header_id|>";
static const char* llamaEot = "<|eot_id|>";
static const char* llamaSystemPreamble =
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n";
static const char* llamaToolsEnvironment = "Environment: ipython\n";
static const char* llamaToolUserPreamble =
    "Given the following functions, please respond with a JSON for a function "
    "call with its proper arguments that best answers the given prompt.\n\n"
    "Respond in the format {\"name\": function name, \"parameters\": "
    "dictionary of argument name and its value}."
    "Do not use variables.\n\n";

namespace {

// Trim ASCII whitespace from both ends, mirroring Jinja's `|trim` filter as
// used by the upstream Llama 3.1 chat template.
std::string trimWhitespace(const std::string& s) {
  static const char* whitespace = " \t\n\r\f\v";
  const auto start = s.find_first_not_of(whitespace);
  if (start == std::string::npos) return "";
  const auto end = s.find_last_not_of(whitespace);
  return s.substr(start, end - start + 1);
}

// Render a tool definition as the upstream chat template does: `t |
// tojson(indent=4)` over the OpenAI envelope `{"type": "function", "function":
// {...}}`.
std::string renderToolJson(const tt::domain::tool_calls::Tool& tool) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "    ";
  return Json::writeString(builder, tool.toJson());
}

}  // namespace

std::string LlamaTokenizer::applyChatTemplate(
    const std::vector<tt::domain::ChatMessage>& messages,
    bool addGenerationPrompt,
    const std::optional<std::vector<tt::domain::tool_calls::Tool>>& tools,
    [[maybe_unused]] bool enableReasoning) const {
  std::ostringstream out;

  // Extract system message; the reference template uses an empty system
  // message when none is provided (no implicit tool-calling preamble).
  std::string systemContent;
  auto remainingMessages = messages;
  if (!messages.empty() && messages[0].role == "system") {
    systemContent = trimWhitespace(messages[0].content);
    remainingMessages = std::vector<tt::domain::ChatMessage>(
        messages.begin() + 1, messages.end());
  }

  const bool hasTools = tools.has_value() && !tools->empty();

  if (cfg_.add_bos_token) out << cfg_.bos_token;

  out << llamaHeaderStart << "system" << llamaHeaderEnd << "\n\n";
  if (hasTools) {
    out << llamaToolsEnvironment;
  }
  out << llamaSystemPreamble << systemContent << llamaEot;

  if (hasTools) {
    if (remainingMessages.empty()) {
      throw std::runtime_error(
          "Cannot put tools in the first user message when there's no first "
          "user message!");
    }
    if (remainingMessages[0].role != "user") {
      throw std::runtime_error(
          "When tools are provided, the first non-system message must have "
          "role='user', but got role='" +
          remainingMessages[0].role + "'");
    }

    const std::string firstUserMessage =
        trimWhitespace(remainingMessages[0].content);
    remainingMessages = std::vector<tt::domain::ChatMessage>(
        remainingMessages.begin() + 1, remainingMessages.end());

    out << llamaHeaderStart << "user" << llamaHeaderEnd << "\n\n";
    out << llamaToolUserPreamble;

    for (const auto& tool : *tools) {
      out << renderToolJson(tool) << "\n\n";
    }

    out << firstUserMessage << llamaEot;
  }

  // Process remaining messages
  for (const auto& m : remainingMessages) {
    if (m.tool_calls.has_value() && !m.tool_calls->empty()) {
      // Assistant message with tool calls
      if (m.tool_calls->size() != 1) {
        throw std::runtime_error(
            "This model only supports single tool-calls at once!");
      }

      const auto& toolCall = (*m.tool_calls)[0];
      out << llamaHeaderStart << "assistant" << llamaHeaderEnd << "\n\n";
      out << "{\"name\": \"" << toolCall.functionCall.name << "\", ";
      out << "\"parameters\": " << toolCall.functionCall.arguments << "}";
      out << llamaEot;

    } else if (m.role == "tool" || m.role == "ipython") {
      // Tool output: the reference template always emits role=`ipython`,
      // regardless of whether the caller used OpenAI's `tool` role or
      // Llama's native `ipython` role.
      out << llamaHeaderStart << "ipython" << llamaHeaderEnd << "\n\n";
      out << trimWhitespace(m.content) << llamaEot;

    } else {
      // Regular user/assistant message
      const std::string role = m.role.empty() ? "user" : m.role;
      out << llamaHeaderStart << role << llamaHeaderEnd << "\n\n";
      out << trimWhitespace(m.content) << llamaEot;
    }
  }

  // Add generation prompt
  if (addGenerationPrompt) {
    out << llamaHeaderStart << "assistant" << llamaHeaderEnd << "\n\n";
  }
  TT_LOG_INFO("LJUBICA LlamaTokenizer::applyChatTemplate: {}", out.str());

  return out.str();
}

}  // namespace tt::utils::tokenizers
