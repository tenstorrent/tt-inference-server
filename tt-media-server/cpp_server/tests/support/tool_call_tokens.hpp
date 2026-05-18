// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// DeepSeek tool call token sequences for integration tests.
// Provides pre-built token ID sequences that produce valid tool call deltas
// when fed through the worker response pipeline.
//
// Usage with WorkerResponse:
//   ToolCallTokens::singleToolCall("get_weather", R"({"location":"SF"})")
//       .sendTo(server->resultQueue());

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ipc/boost/boost_result_queue.hpp"
#include "ipc/interface/result_queue.hpp"

namespace tt::test {

// DeepSeek R1 special token IDs for tool calling
namespace deepseek_tokens {
constexpr uint64_t TOOL_CALLS_BEGIN = 128806;  // <｜tool▁calls▁begin｜>
constexpr uint64_t TOOL_CALLS_END = 128807;    // <｜tool▁calls▁end｜>
constexpr uint64_t TOOL_CALL_BEGIN = 128808;   // <｜tool▁call▁begin｜>
constexpr uint64_t TOOL_CALL_END = 128809;     // <｜tool▁call▁end｜>
constexpr uint64_t TOOL_SEP = 128814;          // <｜tool▁sep｜>
}  // namespace deepseek_tokens

// Fluent builder for tool call token sequences.
// Tokens are encoded as DeepSeek R1 token IDs.
class ToolCallTokens {
 public:
  explicit ToolCallTokens(uint32_t taskId) : taskId_(taskId) {}

  // Build a token sequence for a single tool call.
  // Produces: <tool_calls_begin> <tool_call_begin> function <tool_sep>
  //           {name}\n ```json\n {arguments}\n ``` <tool_call_end>
  //           <tool_calls_end>
  static ToolCallTokens singleToolCall(uint32_t taskId,
                                       const std::string& functionName,
                                       const std::string& arguments) {
    ToolCallTokens builder(taskId);
    builder.beginToolCalls();
    builder.beginToolCall();
    builder.functionType();     // "function"
    builder.toolSeparator();    // <tool_sep>
    builder.text(functionName + "\n");
    builder.text("```json\n");
    builder.text(arguments + "\n");
    builder.text("```\n");
    builder.endToolCall();
    builder.endToolCalls();
    builder.finalize();
    return builder;
  }

  // Build a token sequence for multiple tool calls
  static ToolCallTokens multiToolCall(
      uint32_t taskId,
      const std::vector<std::pair<std::string, std::string>>& calls) {
    ToolCallTokens builder(taskId);
    builder.beginToolCalls();
    for (const auto& [name, args] : calls) {
      builder.beginToolCall();
      builder.functionType();
      builder.toolSeparator();
      builder.text(name + "\n");
      builder.text("```json\n");
      builder.text(args + "\n");
      builder.text("```\n");
      builder.endToolCall();
    }
    builder.endToolCalls();
    builder.finalize();
    return builder;
  }

  // Low-level builders for custom sequences
  ToolCallTokens& beginToolCalls() {
    tokens_.push_back({deepseek_tokens::TOOL_CALLS_BEGIN, 0});
    return *this;
  }

  ToolCallTokens& endToolCalls() {
    tokens_.push_back({deepseek_tokens::TOOL_CALLS_END, 0});
    return *this;
  }

  ToolCallTokens& beginToolCall() {
    tokens_.push_back({deepseek_tokens::TOOL_CALL_BEGIN, 0});
    return *this;
  }

  ToolCallTokens& endToolCall() {
    tokens_.push_back({deepseek_tokens::TOOL_CALL_END, 0});
    return *this;
  }

  ToolCallTokens& toolSeparator() {
    tokens_.push_back({deepseek_tokens::TOOL_SEP, 0});
    return *this;
  }

  ToolCallTokens& functionType() {
    // "function" is typically tokenized, but for mock we use a placeholder
    // The parser looks for the separator, not the actual "function" text
    tokens_.push_back({12345, 0});  // placeholder token for "function"
    return *this;
  }

  // Add text content (uses placeholder token IDs since the actual text
  // is what matters for the parser, not the token ID for non-special tokens)
  ToolCallTokens& text(const std::string& content) {
    // For integration tests, we store the text content and use sequential
    // placeholder IDs. The LLMService will decode these via the tokenizer.
    // Since we're using mock, we need to encode the text to get real token IDs.
    textSegments_.push_back(content);
    tokens_.push_back({static_cast<uint64_t>(20000 + textSegments_.size()), 0});
    return *this;
  }

  // Mark the final token
  ToolCallTokens& finalize() {
    if (!tokens_.empty()) {
      tokens_.back().second |= ipc::SharedToken::FLAG_FINAL;
    }
    return *this;
  }

  // Send tokens to result queue
  void sendTo(ipc::boost::ResultQueue& queue) const {
    uint32_t idx = 0;
    for (const auto& [tokenId, flags] : tokens_) {
      ipc::SharedToken tok{};
      tok.task_id = taskId_;
      tok.token_index = idx++;
      tok.token_id = tokenId;
      tok.flags = flags;
      queue.push(tok);
    }
  }

  // Get raw tokens for inspection
  const std::vector<std::pair<uint64_t, uint32_t>>& tokens() const {
    return tokens_;
  }

  // Get text segments for custom encoding
  const std::vector<std::string>& textSegments() const {
    return textSegments_;
  }

 private:
  uint32_t taskId_;
  std::vector<std::pair<uint64_t, uint32_t>> tokens_;  // {tokenId, flags}
  std::vector<std::string> textSegments_;
};

}  // namespace tt::test
