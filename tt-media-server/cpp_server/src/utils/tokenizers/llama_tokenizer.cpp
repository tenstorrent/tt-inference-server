// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/llama_tokenizer.hpp"

#include <json/json.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace tt::utils::tokenizers {

static const char* llamaHeaderStart = "<|start_header_id|>";
static const char* llamaHeaderEnd = "<|end_header_id|>";
static const char* llamaEot = "<|eot_id|>";
static const char* llamaEnvironmentLine = "Environment: ipython\n";
static const char* llamaDateLines =
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n";
static const char* llamaDefaultToolSystemMessage =
    "You are a helpful assistant with tool calling capabilities. Only reply "
    "with a tool call if the function exists in the library provided by the "
    "user. If it doesn't exist, just reply directly in natural language. When "
    "you receive a tool call response, use the output to format an answer to "
    "the original user question.";
static const char* llamaToolsInUserMessagePrefix =
    "Given the following functions, please respond with a JSON for a function "
    "call with its proper arguments that best answers the given prompt.\n\n"
    "Respond in the format {\"name\": function name, \"parameters\": "
    "dictionary of argument name and its value}. Do not use variables.\n\n";

static std::string trim(const std::string& s) {
  const char* whitespace = " \t\n\r\f\v";
  const size_t start = s.find_first_not_of(whitespace);
  if (start == std::string::npos) return "";
  const size_t end = s.find_last_not_of(whitespace);
  return s.substr(start, end - start + 1);
}

static std::string writeJson(const Json::Value& value,
                             const std::string& indentation) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = indentation;
  return Json::writeString(builder, value);
}

static std::string renderToolCallArguments(const Json::Value& arguments) {
  if (arguments.isString()) {
    Json::Value parsed;
    Json::CharReaderBuilder reader;
    std::string errs;
    std::istringstream stream(arguments.asString());
    if (Json::parseFromStream(reader, stream, &parsed, &errs)) {
      return writeJson(parsed, "");
    }
    return writeJson(arguments, "");
  }
  return writeJson(arguments, "");
}

std::string LlamaTokenizer::applyChatTemplate(
    const std::vector<tt::domain::ChatMessage>& messages,
    bool addGenerationPrompt,
    const std::optional<std::vector<tt::domain::tool_calls::Tool>>& tools)
    const {
  const bool hasTools = tools.has_value() && !tools->empty();

  std::string systemContent;
  bool hadSystemMessage = false;
  std::vector<const tt::domain::ChatMessage*> rest;
  rest.reserve(messages.size());
  for (const auto& m : messages) {
    if (m.role == "system") {
      hadSystemMessage = true;
      if (!systemContent.empty()) systemContent += "\n\n";
      systemContent += trim(m.content);
    } else {
      rest.push_back(&m);
    }
  }
  if (!hadSystemMessage && hasTools) {
    systemContent = llamaDefaultToolSystemMessage;
  }

  std::ostringstream out;
  if (cfg_.add_bos_token) out << cfg_.bos_token;

  out << llamaHeaderStart << "system" << llamaHeaderEnd << "\n\n";
  if (hasTools) out << llamaEnvironmentLine;
  out << llamaDateLines << systemContent << llamaEot;

  size_t messageStartIdx = 0;
  if (hasTools) {
    if (rest.empty()) {
      throw std::runtime_error(
          "Cannot put tools in the first user message when there's no first "
          "user message!");
    }
    out << llamaHeaderStart << "user" << llamaHeaderEnd << "\n\n"
        << llamaToolsInUserMessagePrefix;
    for (const auto& tool : *tools) {
      out << writeJson(tool.toJson(), "    ") << "\n\n";
    }
    out << trim(rest[0]->content) << llamaEot;
    messageStartIdx = 1;
  }

  for (size_t i = messageStartIdx; i < rest.size(); ++i) {
    const auto& m = *rest[i];
    const bool isToolResponse = (m.role == "tool" || m.role == "ipython");
    const bool hasToolCalls =
        m.tool_calls.has_value() && !m.tool_calls->empty();

    if (hasToolCalls) {
      if (m.tool_calls->size() != 1) {
        throw std::runtime_error(
            "This model only supports single tool-calls at once!");
      }
      const auto& call = m.tool_calls->front().functionCall;
      out << llamaHeaderStart << "assistant" << llamaHeaderEnd << "\n\n"
          << "{\"name\": \"" << call.name << "\", "
          << "\"parameters\": " << renderToolCallArguments(call.arguments)
          << "}" << llamaEot;
    } else if (isToolResponse) {
      Json::Value wrapped;
      wrapped["output"] = m.content;
      out << llamaHeaderStart << "ipython" << llamaHeaderEnd << "\n\n"
          << writeJson(wrapped, "") << llamaEot;
    } else {
      const std::string role = m.role.empty() ? "user" : m.role;
      out << llamaHeaderStart << role << llamaHeaderEnd << "\n\n"
          << trim(m.content) << llamaEot;
    }
  }

  if (addGenerationPrompt) {
    out << llamaHeaderStart << "assistant" << llamaHeaderEnd << "\n\n";
  }
  return out.str();
}

}  // namespace tt::utils::tokenizers
