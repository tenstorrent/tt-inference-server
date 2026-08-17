// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Fluent builder for /v1/chat/completions request bodies. Each role-method
// appends a message and returns *this so chains read like a conversation:
//
//   ChatRequest()
//       .system("you are helpful")
//       .user("hi")
//       .assistant("hi back")
//       .user("how are you")
//       .maxTokens(8);
//
// Render with toJson(). Copy a builder to extend it across turns:
//
//   auto t0 = ChatRequest().user("hello");
//   auto t1 = ChatRequest(t0).assistant("ok").user("how are you");

#pragma once

#include <json/json.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tt::test {

class ChatRequest {
 public:
  ChatRequest& system(std::string content) {
    return addMessage("system", std::move(content));
  }
  ChatRequest& user(std::string content) {
    return addMessage("user", std::move(content));
  }
  ChatRequest& assistant(std::string content) {
    return addMessage("assistant", std::move(content));
  }

  ChatRequest& model(std::string m) {
    model_ = std::move(m);
    return *this;
  }
  ChatRequest& maxTokens(int n) {
    maxTokens_ = n;
    return *this;
  }
  ChatRequest& temperature(double t) {
    temperature_ = t;
    return *this;
  }
  ChatRequest& stream(bool on = true) {
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
    if (maxTokens_) root["max_tokens"] = *maxTokens_;
    if (temperature_) root["temperature"] = *temperature_;
    if (stream_) root["stream"] = true;

    Json::StreamWriterBuilder w;
    w["indentation"] = "";
    return Json::writeString(w, root);
  }

 private:
  ChatRequest& addMessage(const char* role, std::string content) {
    messages_.push_back({role, std::move(content)});
    return *this;
  }

  struct Message {
    std::string role;
    std::string content;
  };
  std::vector<Message> messages_;
  std::string model_ = "test";
  std::optional<int> maxTokens_;
  std::optional<double> temperature_;
  bool stream_ = false;
};

}  // namespace tt::test
