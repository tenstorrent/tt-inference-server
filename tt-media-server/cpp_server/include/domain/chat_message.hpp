// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <optional>
#include <string>

#include "tool_calls/tool_call.hpp"

namespace tt::domain {

/** OpenAI chat message: role + content (content may be string or array of
 * parts). */
struct ChatMessage {
  std::string role;
  std::string content;

  std::optional<std::vector<tool_calls::ToolCall>> tool_calls;
  std::optional<std::string> tool_call_id;

  static ChatMessage fromJson(const Json::Value& json) {
    ChatMessage msg;
    if (json.isMember("role") && !json["role"].isNull())
      msg.role = json["role"].asString();
    if (json.isMember("content") && !json["content"].isNull()) {
      const auto& c = json["content"];
      if (c.isString())
        msg.content = c.asString();
      else if (c.isArray())
        for (const auto& part : c)
          if (part.isObject() && part.isMember("type") &&
              part["type"].asString() == "text" && part.isMember("text")) {
            if (!msg.content.empty()) msg.content += ' ';
            msg.content += part["text"].asString();
          }
    }
    if (json.isMember("tool_calls") && json["tool_calls"].isArray()) {
      std::vector<tool_calls::ToolCall> toolCalls;
      for (const auto& toolCall : json["tool_calls"]) {
        toolCalls.push_back(tool_calls::ToolCall::fromJson(toolCall));
      }
      msg.tool_calls = std::move(toolCalls);
    }
    if (json.isMember("tool_call_id") && !json["tool_call_id"].isNull()) {
      msg.tool_call_id = json["tool_call_id"].asString();
    }
    return msg;
  }

  Json::Value toJson() const {
    Json::Value json;
    json["role"] = role;
    json["content"] = content;
    if (tool_calls.has_value() && !tool_calls->empty()) {
      Json::Value toolCallsArray(Json::arrayValue);
      for (const auto& toolCall : *tool_calls) {
        toolCallsArray.append(toolCall.toJson());
      }
      json["tool_calls"] = std::move(toolCallsArray);
    }
    if (tool_call_id.has_value()) {
      json["tool_call_id"] = tool_call_id.value();
    }
    return json;
  }
};

}  // namespace tt::domain
