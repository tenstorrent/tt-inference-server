// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Extends ChatRequest with tools and tool_choice support for tool calling
// integration tests.
//
//   ToolCallRequest()
//       .user("What's the weather in SF?")
//       .tool("get_weather", "Get current weather", {{"location", "string"}})
//       .toolChoice("auto")
//       .maxTokens(128)
//       .stream();

#pragma once

#include <json/json.h>

#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tt::test {

class ToolCallRequest {
 public:
  ToolCallRequest& system(std::string content) {
    return addMessage("system", std::move(content));
  }
  ToolCallRequest& user(std::string content) {
    return addMessage("user", std::move(content));
  }
  ToolCallRequest& assistant(std::string content) {
    return addMessage("assistant", std::move(content));
  }

  // Add a tool with the given function schema
  ToolCallRequest& tool(const std::string& name, const std::string& description,
                        std::initializer_list<std::pair<std::string, std::string>>
                            properties) {
    Json::Value tool;
    tool["type"] = "function";
    tool["function"]["name"] = name;
    tool["function"]["description"] = description;

    Json::Value params;
    params["type"] = "object";
    Json::Value props;
    Json::Value required(Json::arrayValue);
    for (const auto& [propName, propType] : properties) {
      props[propName]["type"] = propType;
      required.append(propName);
    }
    params["properties"] = props;
    params["required"] = required;
    tool["function"]["parameters"] = params;

    tools_.push_back(std::move(tool));
    return *this;
  }

  // Set tool_choice to "auto", "none", "required", or "function"
  ToolCallRequest& toolChoice(const std::string& choice) {
    if (choice == "auto" || choice == "none" || choice == "required") {
      toolChoiceType_ = choice;
      toolChoiceFunction_.reset();
    }
    return *this;
  }

  // Set tool_choice to a specific function name
  ToolCallRequest& toolChoiceFunction(const std::string& functionName) {
    toolChoiceType_ = "function";
    toolChoiceFunction_ = functionName;
    return *this;
  }

  ToolCallRequest& model(std::string m) {
    model_ = std::move(m);
    return *this;
  }
  ToolCallRequest& maxTokens(int n) {
    maxTokens_ = n;
    return *this;
  }
  ToolCallRequest& stream(bool on = true) {
    stream_ = on;
    return *this;
  }

  std::string toJson() const {
    Json::Value root;
    root["model"] = model_;

    Json::Value msgs(Json::arrayValue);
    for (const auto& m : messages_) {
      Json::Value msg;
      msg["role"] = m.role;
      msg["content"] = m.content;
      msgs.append(std::move(msg));
    }
    root["messages"] = std::move(msgs);

    if (!tools_.empty()) {
      Json::Value toolsArray(Json::arrayValue);
      for (const auto& t : tools_) {
        toolsArray.append(t);
      }
      root["tools"] = std::move(toolsArray);
    }

    if (toolChoiceType_.has_value()) {
      if (toolChoiceType_.value() == "function" &&
          toolChoiceFunction_.has_value()) {
        Json::Value tc;
        tc["type"] = "function";
        tc["function"]["name"] = toolChoiceFunction_.value();
        root["tool_choice"] = std::move(tc);
      } else {
        root["tool_choice"] = toolChoiceType_.value();
      }
    }

    if (maxTokens_) root["max_tokens"] = *maxTokens_;
    if (stream_) root["stream"] = true;

    Json::StreamWriterBuilder w;
    w["indentation"] = "";
    return Json::writeString(w, root);
  }

 private:
  ToolCallRequest& addMessage(const char* role, std::string content) {
    messages_.push_back({role, std::move(content)});
    return *this;
  }

  struct Message {
    std::string role;
    std::string content;
  };
  std::vector<Message> messages_;
  std::vector<Json::Value> tools_;
  std::optional<std::string> toolChoiceType_;
  std::optional<std::string> toolChoiceFunction_;
  std::string model_ = "test";
  std::optional<int> maxTokens_;
  bool stream_ = false;
};

}  // namespace tt::test
