// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Wraps the parsed SSE events of an OpenAI-compatible /v1/chat/completions
// streaming response. Lets tests assert on chat-completion-level semantics
// (content deltas, finish_reason, role) without re-parsing JSON inline.
//
//   const auto stream = ChatCompletionStream::parse(response);
//   EXPECT_TRUE(stream.endedWithDone());
//   EXPECT_EQ(stream.initialRole(), "assistant");
//   EXPECT_FALSE(stream.contentDeltas().empty());

#pragma once

#include <json/json.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "support/http_response.hpp"

namespace tt::test {

class ChatCompletionStream {
 public:
  static ChatCompletionStream parse(const HttpResponse& response) {
    ChatCompletionStream s;
    for (const auto& ev : response.sseEvents()) {
      if (ev == "[DONE]") {
        s.endedWithDone_ = true;
        continue;
      }
      Json::Value chunk;
      if (Json::Reader().parse(ev, chunk)) {
        s.chunks_.push_back(std::move(chunk));
      }
    }
    return s;
  }

  bool endedWithDone() const { return endedWithDone_; }

  // Every non-empty content delta in stream order.
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

  // Concatenation of every contentDelta — the full assistant message.
  std::string content() const {
    std::string out;
    for (const auto& d : contentDeltas()) out += d;
    return out;
  }

  // Role announced in the first delta that carries one (usually "assistant").
  std::optional<std::string> initialRole() const {
    for (const auto& c : chunks_) {
      const auto& delta = c["choices"][0]["delta"];
      if (delta.isMember("role") && !delta["role"].asString().empty()) {
        return delta["role"].asString();
      }
    }
    return std::nullopt;
  }

  // finish_reason set on the final chunk that carries one (e.g. "stop",
  // "length"). nullopt if the stream cut off before a finish_reason arrived.
  std::optional<std::string> finishReason() const {
    for (auto it = chunks_.rbegin(); it != chunks_.rend(); ++it) {
      const auto& fr = (*it)["choices"][0]["finish_reason"];
      if (!fr.isNull()) return fr.asString();
    }
    return std::nullopt;
  }

  size_t chunkCount() const { return chunks_.size(); }

 private:
  std::vector<Json::Value> chunks_;
  bool endedWithDone_ = false;
};

}  // namespace tt::test
