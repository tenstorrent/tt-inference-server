// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "utils/llama_tokenizer.hpp"

#include <json/json.h>

#include <sstream>

#include "utils/logger.hpp"

namespace tt::utils {

static const char* llamaHeaderStart = "<|start_header_id|>";
static const char* llamaHeaderEnd = "<|end_header_id|>";
static const char* llamaEot = "<|eot_id|>";
static const char* llamaSystemPreamble =
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n";

std::string LlamaTokenizer::applyChatTemplate(
    const std::vector<domain::ChatMessage>& messages,
    bool addGenerationPrompt,
    const std::optional<std::vector<domain::Tool>>& tools) const {
  std::ostringstream out;

  // Collect system messages
  std::string systemContent;
  for (const auto& m : messages) {
    if (m.role == "system") {
      if (!systemContent.empty()) systemContent += "\n\n";
      systemContent += m.content;
    }
  }

  // Build the system message
  if (cfg_.add_bos_token) out << cfg_.bos_token;

  out << llamaHeaderStart << "system" << llamaHeaderEnd << "\n\n"
  <<llamaSystemPreamble ;

  // Add tool calling capabilities if tools are provided
  if (tools.has_value() && !tools->empty()) {
    out << "You are a helpful assistant with tool-calling capabilities. \n"
        << "When a user asks a question that requires a tool, respond with a "
        << "JSON object inside <tool_call> tags.\n\n";

    out << "Available tools:\n";

    // Build full tools JSON array with type wrapper
    Json::Value toolsArray(Json::arrayValue);
    for (const auto& tool : tools.value()) {
      Json::Value toolWrapper;
      toolWrapper["type"] = tool.type;

      Json::Value functionJson;
      functionJson["name"] = tool.function.name;
      if (tool.function.description.has_value()) {
        functionJson["description"] = tool.function.description.value();
      }
      if (!tool.function.parameters.isNull()) {
        functionJson["parameters"] = tool.function.parameters;
      }
      if (tool.function.strict.has_value()) {
        functionJson["strict"] = tool.function.strict.value();
      }

      toolWrapper["function"] = functionJson;
      toolsArray.append(toolWrapper);
    }

    // Serialize tools array as pretty JSON
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "  ";
    writer["emitUTF8"] = true;
    writer["commentStyle"] = "None";
    writer["enableYAMLCompatibility"] = false;
    writer["dropNullPlaceholders"] = false;
    writer["useSpecialFloats"] = false;
    writer["precision"] = 17;
    out << Json::writeString(writer, toolsArray);
  } else if (!systemContent.empty()) {
    out << systemContent;
  }

  out << llamaEot;

  // Add non-system messages
  for (const auto& m : messages) {
    if (m.role == "system") continue;
    std::string role = m.role.empty() ? "user" : m.role;
    out << llamaHeaderStart << role << llamaHeaderEnd << "\n\n"
        << m.content << llamaEot;
  }

  if (addGenerationPrompt) {
    out << llamaHeaderStart << "assistant" << llamaHeaderEnd << "\n\n";
  }

  std::string result = out.str();

  TT_LOG_DEBUG("[LlamaTokenizer] Final prompt:\n{}", result);

  return result;
}

}  // namespace tt::utils
