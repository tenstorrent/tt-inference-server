// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/hf_cpu_tokenizer.hpp"

#include <json/writer.h>

#include <cstdlib>
#include <sstream>

namespace tt::utils::tokenizers {

std::string HFCPUTokenizer::modelName() const {
  // Return the model name from env or a generic default
  const char* model = std::getenv("HF_MODEL");
  if (model && *model) {
    return model;
  }
  return "hf-cpu-model";
}

std::vector<int64_t> HFCPUTokenizer::stopTokenIds() const {
  // Read stop token IDs from env HF_STOP_TOKEN_IDS
  const char* env = std::getenv("HF_STOP_TOKEN_IDS");
  if (env && *env) {
    std::vector<int64_t> ids;
    std::string s(env);
    size_t pos = 0;
    while (pos < s.size()) {
      size_t comma = s.find(',', pos);
      std::string token = s.substr(pos, comma - pos);
      // Trim whitespace
      size_t start = token.find_first_not_of(" \t");
      size_t end = token.find_last_not_of(" \t");
      if (start != std::string::npos && end != std::string::npos) {
        ids.push_back(std::stoll(token.substr(start, end - start + 1)));
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    if (!ids.empty()) {
      return ids;
    }
  }
  return {};
}

namespace {

std::string serializeToolsBlock(
    const std::vector<tt::domain::tool_calls::Tool>& tools) {
  Json::StreamWriterBuilder writerBuilder;
  writerBuilder["indentation"] = "";
  std::ostringstream out;
  for (const auto& tool : tools) {
    out << "\n" << Json::writeString(writerBuilder, tool.toJson());
  }
  return out.str();
}

}  // namespace

std::string HFCPUTokenizer::applyChatTemplate(
    const std::vector<tt::domain::llm::ChatMessage>& messages,
    bool addGenerationPrompt,
    const std::optional<std::vector<tt::domain::tool_calls::Tool>>& tools,
    [[maybe_unused]] bool enableReasoning, bool skipApplyChatTemplate) const {
  if (skipApplyChatTemplate) {
    std::ostringstream out;
    for (const auto& m : messages) {
      out << m.content;
    }
    return out.str();
  }

  // Qwen-compatible chat template. When `tools` is provided we inline the
  // Qwen tool-calling system prompt that the jinja template in
  // tokenizers/Qwen/Qwen2.5-1.5B-Instruct/chat_template.jinja produces.
  const bool hasTools = tools.has_value() && !tools->empty();

  std::ostringstream out;
  size_t i = 0;

  // Build the leading system message (with optional tools section).
  std::string systemContent;
  bool consumedFirstSystem = false;
  if (!messages.empty() && messages.front().role == "system") {
    systemContent = messages.front().content;
    consumedFirstSystem = true;
  } else if (hasTools) {
    systemContent =
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
  }

  if (hasTools) {
    out << "<|im_start|>system\n";
    out << systemContent;
    out << "\n\n# Tools\n\n"
        << "You may call one or more functions to assist with the user "
           "query.\n\n"
        << "You are provided with function signatures within "
           "<tools></tools> XML tags:\n"
        << "<tools>" << serializeToolsBlock(*tools) << "\n</tools>\n\n"
        << "For each function call, return a json object with function name "
           "and arguments within <tool_call></tool_call> XML tags:\n"
        << "<tool_call>\n"
        << "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
        << "</tool_call><|im_end|>\n";
  } else if (consumedFirstSystem) {
    out << "<|im_start|>system\n" << systemContent << "<|im_end|>\n";
  }

  if (consumedFirstSystem) i = 1;

  for (; i < messages.size(); ++i) {
    const auto& m = messages[i];
    if (m.role == "system") {
      out << "<|im_start|>system\n" << m.content << "<|im_end|>\n";
    } else if (m.role == "user") {
      out << "<|im_start|>user\n" << m.content << "<|im_end|>\n";
    } else if (m.role == "assistant") {
      out << "<|im_start|>assistant\n" << m.content << "<|im_end|>\n";
    } else if (m.role == "tool") {
      out << "<|im_start|>user\n<tool_response>\n"
          << m.content << "\n</tool_response><|im_end|>\n";
    }
  }
  if (addGenerationPrompt) {
    out << "<|im_start|>assistant\n";
  }
  return out.str();
}

}  // namespace tt::utils::tokenizers
