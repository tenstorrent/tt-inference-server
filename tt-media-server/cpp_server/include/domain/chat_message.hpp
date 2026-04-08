// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <string>

namespace tt::domain {

/** OpenAI chat message: role + content (content may be string or array of
 * parts). */
struct ChatMessage {
  std::string role;
  std::string content;

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
    return msg;
  }
};

}  // namespace tt::domain
