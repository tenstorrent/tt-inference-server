// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/llama_tokenizer.hpp"

#include <json/json.h>

#include <sstream>
#include <stdexcept>

namespace tt::utils::tokenizers {

static const char* llamaHeaderStart = "<|start_header_id|>";
static const char* llamaHeaderEnd = "<|end_header_id|>";
static const char* llamaEot = "<|eot_id|>";
static const char* llamaSystemPreamble =
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n";

std::string LlamaTokenizer::applyChatTemplate(
    const std::vector<tt::domain::ChatMessage>& messages,
    bool addGenerationPrompt,
    const std::optional<std::vector<tt::domain::tool_calls::Tool>>& tools) const {
  std::ostringstream out;

  // Helper to serialize tool to JSON string
  auto toolToJsonString = [](const tt::domain::tool_calls::Tool& tool) {
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "    ";
    return Json::writeString(builder, tool.toJson());
  };

  // Extract system message
  std::string systemContent;
  auto filteredMessages = messages;
  if (!messages.empty() && messages[0].role == "system") {
    systemContent = messages[0].content;
    filteredMessages = std::vector<tt::domain::ChatMessage>(
        messages.begin() + 1, messages.end());
  } else if (tools.has_value() && !tools->empty()) {
    // Default system message when tools are provided
    systemContent = "You are a helpful assistant with tool calling capabilities. "
                    "Only reply with a tool call if the function exists in the library "
                    "provided by the user. If it doesn't exist, just reply directly in "
                    "natural language. When you receive a tool call response, use the "
                    "output to format an answer to the original user question.";
  }

  // BOS token
  if (cfg_.add_bos_token) out << cfg_.bos_token;

  // System header
  out << llamaHeaderStart << "system" << llamaHeaderEnd << "\n\n";

  // Add environment for tools
  if (tools.has_value() && !tools->empty()) {
    out << "Environment: ipython\n";
  }

  out << llamaSystemPreamble;

  // Add tools in system message if toolsInUserMessage is false
  // For Llama 3.1, we default to toolsInUserMessage=true
  bool toolsInUserMessage = true;

  if (tools.has_value() && !tools->empty() && !toolsInUserMessage) {
    out << "You have access to the following functions. To call a function, "
        << "please respond with JSON for a function call. "
        << "Respond in the format {\"name\": function name, \"parameters\": "
        << "dictionary of argument name and its value}. "
        << "Do not use variables.\n\n";

    for (const auto& tool : *tools) {
      out << toolToJsonString(tool) << "\n\n";
    }
  }

  out << systemContent << llamaEot;

  // Add tools in user message if toolsInUserMessage is true
  if (tools.has_value() && !tools->empty() && toolsInUserMessage) {
    if (filteredMessages.empty()) {
      throw std::runtime_error(
          "Cannot put tools in the first user message when there's no first user message!");
    }

    // Extract first user message
    auto firstUserMessage = filteredMessages[0].content;
    filteredMessages = std::vector<tt::domain::ChatMessage>(
        filteredMessages.begin() + 1, filteredMessages.end());

    out << llamaHeaderStart << "user" << llamaHeaderEnd << "\n\n";
    out << "Given the following functions, please respond with a JSON for a function call "
        << "with its proper arguments that best answers the given prompt.\n\n"
        << "Respond in the format {\"name\": function name, \"parameters\": "
        << "dictionary of argument name and its value}. "
        << "Do not use variables.\n\n";

    for (const auto& tool : *tools) {
      out << toolToJsonString(tool) << "\n\n";
    }

    out << firstUserMessage << llamaEot;
  }

  // Process remaining messages
  for (const auto& m : filteredMessages) {
    // Skip messages that have tool_calls, tool role, or ipython role
    // They need special handling
    if (m.tool_calls.has_value() && !m.tool_calls->empty()) {
      // Assistant message with tool calls
      if (m.tool_calls->size() != 1) {
        throw std::runtime_error("This model only supports single tool-calls at once!");
      }

      const auto& toolCall = (*m.tool_calls)[0];
      out << llamaHeaderStart << "assistant" << llamaHeaderEnd << "\n\n";
      out << "{\"name\": \"" << toolCall.functionCall.name << "\", ";
      out << "\"parameters\": ";

      // Serialize arguments as JSON
      Json::StreamWriterBuilder builder;
      builder["indentation"] = "";
      out << Json::writeString(builder, toolCall.functionCall.arguments);
      out << "}" << llamaEot;

    } else if (m.role == "tool" || m.role == "ipython") {
      // Tool output message
      out << llamaHeaderStart << "ipython" << llamaHeaderEnd << "\n\n";
      out << m.content << llamaEot;

    } else {
      // Regular user/assistant message
      std::string role = m.role.empty() ? "user" : m.role;
      out << llamaHeaderStart << role << llamaHeaderEnd << "\n\n";
      out << m.content << llamaEot;
    }
  }

  // Add generation prompt
  if (addGenerationPrompt) {
    out << llamaHeaderStart << "assistant" << llamaHeaderEnd << "\n\n";
  }

  return out.str();
}

}  // namespace tt::utils::tokenizers
