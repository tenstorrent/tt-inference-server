// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Wraps parsed SSE events from a streaming response that includes tool_calls.
// Provides semantic accessors for tool call deltas in addition to regular
// content.
//
//   const auto stream = ToolCallStream::parse(response);
//   EXPECT_EQ(stream.finishReason(), "tool_calls");
//   EXPECT_EQ(stream.toolCallCount(), 1);
//   EXPECT_EQ(stream.toolCallFunctionName(0), "get_weather");

#pragma once

#include <json/json.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "support/http_response.hpp"

namespace tt::test {

// Parsed tool call from streamed deltas
struct ParsedToolCall {
  std::string id;
  std::string type;
  std::string functionName;
  std::string arguments;
};

class ToolCallStream {
 public:
  static ToolCallStream parse(const HttpResponse& response) {
    ToolCallStream s;
    for (const auto& ev : response.sseEvents()) {
      if (ev == "[DONE]") {
        s.endedWithDone_ = true;
        continue;
      }
      Json::Value chunk;
      if (Json::Reader().parse(ev, chunk)) {
        s.chunks_.push_back(chunk);
        s.processChunk(chunk);
      }
    }
    s.finalizeToolCalls();
    return s;
  }

  bool endedWithDone() const { return endedWithDone_; }

  // Role announced in the first delta (usually "assistant")
  std::optional<std::string> initialRole() const {
    for (const auto& c : chunks_) {
      const auto& delta = c["choices"][0]["delta"];
      if (delta.isMember("role") && !delta["role"].asString().empty()) {
        return delta["role"].asString();
      }
    }
    return std::nullopt;
  }

  // Content deltas (regular text)
  std::vector<std::string> contentDeltas() const {
    std::vector<std::string> out;
    for (const auto& c : chunks_) {
      const auto& delta = c["choices"][0]["delta"];
      if (delta.isMember("content") && !delta["content"].asString().empty()) {
        out.push_back(delta["content"].asString());
      }
    }
    return out;
  }

  // Full content string
  std::string content() const {
    std::string out;
    for (const auto& d : contentDeltas()) out += d;
    return out;
  }

  // finish_reason from the final chunk
  std::optional<std::string> finishReason() const {
    for (auto it = chunks_.rbegin(); it != chunks_.rend(); ++it) {
      const auto& fr = (*it)["choices"][0]["finish_reason"];
      if (!fr.isNull()) return fr.asString();
    }
    return std::nullopt;
  }

  // Number of tool calls parsed
  size_t toolCallCount() const { return toolCalls_.size(); }

  // Get a specific tool call
  const ParsedToolCall& toolCall(size_t index) const {
    return toolCalls_.at(index);
  }

  // Convenience accessors
  std::string toolCallId(size_t index) const { return toolCalls_.at(index).id; }

  std::string toolCallFunctionName(size_t index) const {
    return toolCalls_.at(index).functionName;
  }

  std::string toolCallArguments(size_t index) const {
    return toolCalls_.at(index).arguments;
  }

  // Check if any tool calls were streamed
  bool hasToolCalls() const { return !toolCalls_.empty(); }

  // All tool call deltas received (for debugging)
  const std::vector<Json::Value>& toolCallDeltas() const {
    return toolCallDeltas_;
  }

  size_t chunkCount() const { return chunks_.size(); }

 private:
  void processChunk(const Json::Value& chunk) {
    if (!chunk["choices"].isArray() || chunk["choices"].empty()) return;

    const auto& delta = chunk["choices"][0]["delta"];
    if (!delta.isMember("tool_calls")) return;

    const auto& toolCallsArray = delta["tool_calls"];
    if (!toolCallsArray.isArray()) return;

    for (const auto& tc : toolCallsArray) {
      toolCallDeltas_.push_back(tc);

      int index = tc["index"].asInt();

      // Ensure we have enough slots
      while (static_cast<size_t>(index) >= pendingToolCalls_.size()) {
        pendingToolCalls_.push_back({});
      }

      auto& pending = pendingToolCalls_[index];

      // Capture id and type from first delta for this index
      if (tc.isMember("id") && !tc["id"].asString().empty()) {
        pending.id = tc["id"].asString();
      }
      if (tc.isMember("type") && !tc["type"].asString().empty()) {
        pending.type = tc["type"].asString();
      }

      // Capture function name and accumulate arguments
      if (tc.isMember("function")) {
        const auto& fn = tc["function"];
        if (fn.isMember("name") && !fn["name"].asString().empty()) {
          pending.functionName = fn["name"].asString();
        }
        if (fn.isMember("arguments")) {
          pending.arguments += fn["arguments"].asString();
        }
      }
    }
  }

  void finalizeToolCalls() {
    for (auto& pending : pendingToolCalls_) {
      if (!pending.functionName.empty()) {
        toolCalls_.push_back(std::move(pending));
      }
    }
  }

  std::vector<Json::Value> chunks_;
  std::vector<Json::Value> toolCallDeltas_;
  std::vector<ParsedToolCall> pendingToolCalls_;
  std::vector<ParsedToolCall> toolCalls_;
  bool endedWithDone_ = false;
};

}  // namespace tt::test
