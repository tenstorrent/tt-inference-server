// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Mock runner for tool calling integration tests.
//
// This runner streams predefined token sequences that represent raw model
// output for tool calls. It allows testing the full pipeline:
//   tokenizer decoding → tool_call_parser → SSE formatting
//
// Usage:
//   MockToolCallRunner runner(server->resultQueue());
//   runner.queueToolCall("get_weather", R"({"location":"SF"})");
//
//   // ... send request ...
//   auto seq = server->taskQueue().receive();
//   runner.streamTo(seq->taskId);  // streams the queued tokens

#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ipc/boost/boost_result_queue.hpp"
#include "ipc/interface/result_queue.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::test {

// DeepSeek R1 special token IDs for tool calling
namespace deepseek {
constexpr uint64_t TOOL_CALLS_BEGIN = 128806;  // <｜tool▁calls▁begin｜>
constexpr uint64_t TOOL_CALLS_END = 128807;    // <｜tool▁calls▁end｜>
constexpr uint64_t TOOL_CALL_BEGIN = 128808;   // <｜tool▁call▁begin｜>
constexpr uint64_t TOOL_CALL_END = 128809;     // <｜tool▁call▁end｜>
constexpr uint64_t TOOL_SEP = 128814;          // <｜tool▁sep｜>
}  // namespace deepseek

class MockToolCallRunner {
 public:
  explicit MockToolCallRunner(ipc::boost::ResultQueue& queue) : queue_(queue) {}

  // Queue a single tool call to be streamed.
  // This builds the DeepSeek format:
  //   <tool_calls_begin><tool_call_begin>function<tool_sep>name\n
  //   ```json\n{args}\n```\n<tool_call_end><tool_calls_end>
  MockToolCallRunner& queueToolCall(const std::string& functionName,
                                    const std::string& arguments) {
    // Start tool calls block
    tokens_.push_back(deepseek::TOOL_CALLS_BEGIN);
    tokens_.push_back(deepseek::TOOL_CALL_BEGIN);

    // "function" before separator
    appendEncoded("function");
    tokens_.push_back(deepseek::TOOL_SEP);

    // Function name with newline
    appendEncoded(functionName + "\n");

    // JSON block
    appendEncoded("```json\n");
    appendEncoded(arguments + "\n");
    appendEncoded("```\n");

    // End markers
    tokens_.push_back(deepseek::TOOL_CALL_END);
    tokens_.push_back(deepseek::TOOL_CALLS_END);

    return *this;
  }

  // Queue multiple tool calls in a single response
  MockToolCallRunner& queueMultiToolCall(
      const std::vector<std::pair<std::string, std::string>>& calls) {
    tokens_.push_back(deepseek::TOOL_CALLS_BEGIN);

    for (const auto& [name, args] : calls) {
      tokens_.push_back(deepseek::TOOL_CALL_BEGIN);
      appendEncoded("function");
      tokens_.push_back(deepseek::TOOL_SEP);
      appendEncoded(name + "\n");
      appendEncoded("```json\n");
      appendEncoded(args + "\n");
      appendEncoded("```\n");
      tokens_.push_back(deepseek::TOOL_CALL_END);
    }

    tokens_.push_back(deepseek::TOOL_CALLS_END);
    return *this;
  }

  // Queue regular text (non-tool-call response)
  MockToolCallRunner& queueText(const std::string& text) {
    appendEncoded(text);
    return *this;
  }

  // Queue text followed by a tool call
  MockToolCallRunner& queueTextThenToolCall(const std::string& text,
                                            const std::string& functionName,
                                            const std::string& arguments) {
    appendEncoded(text);
    queueToolCall(functionName, arguments);
    return *this;
  }

  // Stream the queued tokens to the result queue for the given task
  void streamTo(uint32_t taskId) {
    for (size_t i = 0; i < tokens_.size(); ++i) {
      ipc::SharedToken tok{};
      tok.task_id = taskId;
      tok.token_index = static_cast<uint32_t>(i);
      tok.token_id = tokens_[i];
      tok.flags = (i == tokens_.size() - 1) ? ipc::SharedToken::FLAG_FINAL : 0;
      queue_.push(tok);
    }
    tokens_.clear();
  }

  // Get the raw token IDs (for debugging)
  const std::vector<uint64_t>& tokens() const { return tokens_; }

  // Clear any queued tokens
  void clear() { tokens_.clear(); }

  // Debug: print tokens and their decoded text
  void debugPrint() const {
    auto& tok = utils::tokenizers::activeTokenizer();
    std::cout << "=== MockToolCallRunner Tokens (" << tokens_.size()
              << ") ===" << std::endl;
    for (size_t i = 0; i < tokens_.size(); ++i) {
      auto decoded = tok.decode({static_cast<int>(tokens_[i])}, false);
      std::cout << "  [" << i << "] " << tokens_[i];
      if (tokens_[i] == deepseek::TOOL_CALLS_BEGIN)
        std::cout << " (TOOL_CALLS_BEGIN)";
      else if (tokens_[i] == deepseek::TOOL_CALLS_END)
        std::cout << " (TOOL_CALLS_END)";
      else if (tokens_[i] == deepseek::TOOL_CALL_BEGIN)
        std::cout << " (TOOL_CALL_BEGIN)";
      else if (tokens_[i] == deepseek::TOOL_CALL_END)
        std::cout << " (TOOL_CALL_END)";
      else if (tokens_[i] == deepseek::TOOL_SEP)
        std::cout << " (TOOL_SEP)";
      else
        std::cout << " -> \"" << decoded << "\"";
      std::cout << std::endl;
    }
  }

 private:
  void appendEncoded(const std::string& text) {
    auto ids = utils::tokenizers::activeTokenizer().encode(text);
    for (int id : ids) {
      tokens_.push_back(static_cast<uint64_t>(id));
    }
  }

  ipc::boost::ResultQueue& queue_;
  std::vector<uint64_t> tokens_;
};

}  // namespace tt::test
