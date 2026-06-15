// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <gtest/gtest.h>

#include <string>

#include "config/types.hpp"
#include "services/tool_call_parser.hpp"

using namespace tt::services;

namespace {

// Token IDs (from DeepSeek tokenizer)
constexpr int64_t kToolCallsBeginToken = 128806;
constexpr int64_t kToolCallsEndToken = 128807;
constexpr int64_t kToolCallBeginToken = 128808;
constexpr int64_t kToolCallEndToken = 128809;
constexpr int64_t kToolSepToken = 128814;

class ToolCallParserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    parser_ = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);
  }

  std::unique_ptr<IToolCallParser> parser_;
};

TEST_F(ToolCallParserTest, StreamingTokens) {
  uint32_t taskId = 1;

  parser_->initializeTask(taskId);
  EXPECT_EQ(parser_->activeTaskCount(), 1);

  // Simulate: <｜tool▁calls▁begin｜>
  {
    auto r = parser_->processToken(taskId, kToolCallsBeginToken, "");
    EXPECT_FALSE(r.has_value());
    EXPECT_TRUE(parser_->isInToolCall(taskId));
  }

  // Simulate: <｜tool▁call▁begin｜>
  {
    auto r = parser_->processToken(taskId, kToolCallBeginToken, "");
    EXPECT_FALSE(r.has_value());
  }

  // Simulate: "function"
  {
    auto r = parser_->processToken(taskId, 12345, "function");
    EXPECT_FALSE(r.has_value());
  }

  // Simulate: <｜tool▁sep｜>
  {
    auto r = parser_->processToken(taskId, kToolSepToken, "");
    EXPECT_FALSE(r.has_value());
  }

  // Simulate: "get_weather\n" - this emits TOOL_CALL_START
  {
    auto r = parser_->processToken(taskId, 12346, "get_weather\n");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->delta_type, ToolCallDeltaType::TOOL_CALL_START);
    EXPECT_EQ(r->function_name, "get_weather");
  }

  // Simulate: "```json\n"
  {
    auto r = parser_->processToken(taskId, 12347, "```json\n");
    EXPECT_FALSE(r.has_value());
  }

  // Simulate: JSON arguments - emits ARGUMENTS_DELTA
  {
    auto r = parser_->processToken(taskId, 12348,
                                   "{\"location\":\"San Francisco\"}\n");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->delta_type, ToolCallDeltaType::ARGUMENTS_DELTA);
  }

  // Simulate: "```\n"
  {
    auto r = parser_->processToken(taskId, 12349, "```\n");
    EXPECT_FALSE(r.has_value());
  }

  // Simulate: <｜tool▁call▁end｜> - emits TOOL_CALL_END
  {
    auto r = parser_->processToken(taskId, kToolCallEndToken, "");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->delta_type, ToolCallDeltaType::TOOL_CALL_END);
  }

  // Simulate: <｜tool▁calls▁end｜>
  {
    auto r = parser_->processToken(taskId, kToolCallsEndToken, "");
    EXPECT_FALSE(r.has_value());
    EXPECT_FALSE(parser_->isInToolCall(taskId));
  }

  // Regular text after tool calls - parser returns nullopt, not in tool call
  {
    auto r = parser_->processToken(taskId, 11111, "The answer is ready.");
    EXPECT_FALSE(r.has_value());
    EXPECT_FALSE(parser_->isInToolCall(taskId));
  }

  // Finalize and check tool calls were parsed
  auto toolCalls = parser_->finalizeTask(taskId);
  ASSERT_TRUE(toolCalls.has_value());
  ASSERT_TRUE(toolCalls->isArray());
  ASSERT_EQ(toolCalls->size(), 1);

  auto& toolCall = (*toolCalls)[0];
  EXPECT_EQ(toolCall["id"].asString(), "call_0");
  EXPECT_EQ(toolCall["type"].asString(), "function");
  EXPECT_EQ(toolCall["function"]["name"].asString(), "get_weather");

  EXPECT_EQ(parser_->activeTaskCount(), 0);
}

TEST_F(ToolCallParserTest, MultipleConcurrentStreamingTasks) {
  // Initialize multiple tasks
  for (uint32_t i = 0; i < 10; ++i) {
    parser_->initializeTask(i);
  }

  EXPECT_EQ(parser_->activeTaskCount(), 10);

  // Process tokens for different tasks in interleaved manner
  for (uint32_t i = 0; i < 10; i += 2) {
    auto r = parser_->processToken(i, kToolCallsBeginToken, "");
    EXPECT_FALSE(r.has_value());
    EXPECT_TRUE(parser_->isInToolCall(i));
  }

  // Check odd tasks are not in tool call mode
  for (uint32_t i = 1; i < 10; i += 2) {
    EXPECT_FALSE(parser_->isInToolCall(i));
  }

  // Finalize all tasks
  for (uint32_t i = 0; i < 10; ++i) {
    parser_->finalizeTask(i);
  }

  EXPECT_EQ(parser_->activeTaskCount(), 0);
}

TEST_F(ToolCallParserTest, UninitializedTaskReturnsNullopt) {
  auto r = parser_->processToken(99999, 12345, "text");
  EXPECT_FALSE(r.has_value());
  EXPECT_FALSE(parser_->isInToolCall(99999));
}

TEST_F(ToolCallParserTest, FinalizeWhileInToolCall) {
  uint32_t taskId = 50;
  parser_->initializeTask(taskId);

  parser_->processToken(taskId, kToolCallsBeginToken, "");
  EXPECT_TRUE(parser_->isInToolCall(taskId));

  auto result = parser_->finalizeTask(taskId);
  EXPECT_EQ(parser_->activeTaskCount(), 0);
}

TEST_F(ToolCallParserTest, RegularContentBeforeAndAfterToolCalls) {
  uint32_t taskId = 51;
  parser_->initializeTask(taskId);

  // Regular text before - returns nullopt, not in tool call
  {
    auto r = parser_->processToken(taskId, 12345, "Let me help you.");
    EXPECT_FALSE(r.has_value());
    EXPECT_FALSE(parser_->isInToolCall(taskId));
  }

  // Tool calls block
  parser_->processToken(taskId, kToolCallsBeginToken, "");
  EXPECT_TRUE(parser_->isInToolCall(taskId));

  parser_->processToken(taskId, kToolCallsEndToken, "");
  EXPECT_FALSE(parser_->isInToolCall(taskId));

  // Regular text after - returns nullopt, not in tool call
  {
    auto r = parser_->processToken(taskId, 12346, "Done.");
    EXPECT_FALSE(r.has_value());
    EXPECT_FALSE(parser_->isInToolCall(taskId));
  }

  parser_->finalizeTask(taskId);
}

TEST_F(ToolCallParserTest, SpuriousToolCallEndBeforeToolCallBegin) {
  uint32_t taskId = 200;
  parser_->initializeTask(taskId);

  // Enter tool calls block
  parser_->processToken(taskId, kToolCallsBeginToken, "");
  EXPECT_TRUE(parser_->isInToolCall(taskId));

  // Spurious tool_call_end without matching begin - should handle gracefully
  parser_->processToken(taskId, kToolCallEndToken, "");

  // Now start a proper tool call
  parser_->processToken(taskId, kToolCallBeginToken, "");
  parser_->processToken(taskId, 12345, "function");
  parser_->processToken(taskId, kToolSepToken, "");
  parser_->processToken(taskId, 12346, "get_weather\n");
  parser_->processToken(taskId, 12347, "```json\n");
  parser_->processToken(taskId, 12348, "{\"location\":\"SF\"}\n");
  parser_->processToken(taskId, 12349, "```\n");
  parser_->processToken(taskId, kToolCallEndToken, "");
  parser_->processToken(taskId, kToolCallsEndToken, "");

  auto toolCalls = parser_->finalizeTask(taskId);
  // Should have only the valid tool call, spurious end was ignored
  ASSERT_TRUE(toolCalls.has_value());
  ASSERT_EQ(toolCalls->size(), 1);
  EXPECT_EQ((*toolCalls)[0]["function"]["name"].asString(), "get_weather");
}

TEST_F(ToolCallParserTest, ToolCallsEndBeforeBegin) {
  uint32_t taskId = 201;
  parser_->initializeTask(taskId);

  // Spurious end token while in REGULAR state
  auto r1 = parser_->processToken(taskId, kToolCallsEndToken, "");
  EXPECT_FALSE(r1.has_value());
  EXPECT_FALSE(parser_->isInToolCall(taskId));

  // Text that should be treated as regular content
  auto r2 = parser_->processToken(taskId, 12345, "some tool name");
  EXPECT_FALSE(r2.has_value());
  EXPECT_FALSE(parser_->isInToolCall(taskId));

  // Now proper begin
  parser_->processToken(taskId, kToolCallsBeginToken, "");
  EXPECT_TRUE(parser_->isInToolCall(taskId));

  parser_->processToken(taskId, kToolCallsEndToken, "");
  EXPECT_FALSE(parser_->isInToolCall(taskId));

  auto toolCalls = parser_->finalizeTask(taskId);
  // No valid tool calls parsed
  EXPECT_TRUE(!toolCalls.has_value() || toolCalls->size() == 0);
}

TEST_F(ToolCallParserTest, DoubleToolCallsBegin) {
  uint32_t taskId = 202;
  parser_->initializeTask(taskId);

  parser_->processToken(taskId, kToolCallsBeginToken, "");
  EXPECT_TRUE(parser_->isInToolCall(taskId));

  // Second begin - should stay in tool call mode
  parser_->processToken(taskId, kToolCallsBeginToken, "");
  EXPECT_TRUE(parser_->isInToolCall(taskId));

  parser_->processToken(taskId, kToolCallsEndToken, "");
  EXPECT_FALSE(parser_->isInToolCall(taskId));

  parser_->finalizeTask(taskId);
}

TEST_F(ToolCallParserTest, ToolCallBeginWithoutOuterWrapper) {
  uint32_t taskId = 203;
  parser_->initializeTask(taskId);

  // Individual tool call markers without outer wrapper
  parser_->processToken(taskId, kToolCallBeginToken, "");
  // State machine goes to IN_TOOL_CALL even without outer wrapper
  EXPECT_TRUE(parser_->isInToolCall(taskId));

  parser_->processToken(taskId, 12345, "function");
  parser_->processToken(taskId, kToolSepToken, "");
  parser_->processToken(taskId, 12346, "get_time\n");
  parser_->processToken(taskId, 12347, "```json\n");
  parser_->processToken(taskId, 12348, "{}\n");
  parser_->processToken(taskId, 12349, "```\n");
  parser_->processToken(taskId, kToolCallEndToken, "");

  auto toolCalls = parser_->finalizeTask(taskId);
  // Should still parse the tool call
  ASSERT_TRUE(toolCalls.has_value());
  ASSERT_EQ(toolCalls->size(), 1);
}

TEST_F(ToolCallParserTest, IncompleteToolCallMissingEnd) {
  uint32_t taskId = 204;
  parser_->initializeTask(taskId);

  parser_->processToken(taskId, kToolCallsBeginToken, "");
  parser_->processToken(taskId, kToolCallBeginToken, "");
  parser_->processToken(taskId, 12345, "function");
  parser_->processToken(taskId, kToolSepToken, "");
  parser_->processToken(taskId, 12346, "get_weather\n");
  parser_->processToken(taskId, 12347, "```json\n");
  parser_->processToken(taskId, 12348, "{\"x\":1}\n");
  // Missing tool_call_end and tool_calls_end - just finalize

  auto toolCalls = parser_->finalizeTask(taskId);
  // Incomplete tool call not finalized to array
  EXPECT_TRUE(!toolCalls.has_value() || toolCalls->size() == 0);
}

TEST_F(ToolCallParserTest, GarbageTextBetweenMarkers) {
  uint32_t taskId = 205;
  parser_->initializeTask(taskId);

  parser_->processToken(taskId, kToolCallsBeginToken, "");
  // Random garbage text between tool_calls_begin and tool_call_begin
  parser_->processToken(taskId, 12345, "random garbage here\n");
  parser_->processToken(taskId, 12346, "more nonsense!!!");

  parser_->processToken(taskId, kToolCallBeginToken, "");
  parser_->processToken(taskId, 12347, "function");
  parser_->processToken(taskId, kToolSepToken, "");
  parser_->processToken(taskId, 12348, "valid_func\n");
  parser_->processToken(taskId, 12349, "```json\n");
  parser_->processToken(taskId, 12350, "{}\n");
  parser_->processToken(taskId, 12351, "```\n");
  parser_->processToken(taskId, kToolCallEndToken, "");
  parser_->processToken(taskId, kToolCallsEndToken, "");

  auto toolCalls = parser_->finalizeTask(taskId);
  ASSERT_TRUE(toolCalls.has_value());
  ASSERT_EQ(toolCalls->size(), 1);
  EXPECT_EQ((*toolCalls)[0]["function"]["name"].asString(), "valid_func");
}

TEST_F(ToolCallParserTest, MultipleConsecutiveToolSepTokens) {
  uint32_t taskId = 206;
  parser_->initializeTask(taskId);

  parser_->processToken(taskId, kToolCallsBeginToken, "");
  parser_->processToken(taskId, kToolCallBeginToken, "");
  parser_->processToken(taskId, 12345, "function");
  parser_->processToken(taskId, kToolSepToken, "");
  parser_->processToken(taskId, kToolSepToken, "");  // Extra sep
  parser_->processToken(taskId, kToolSepToken, "");  // Another extra
  parser_->processToken(taskId, 12346, "my_func\n");
  parser_->processToken(taskId, 12347, "```json\n");
  parser_->processToken(taskId, 12348, "{\"key\":\"value\"}\n");
  parser_->processToken(taskId, 12349, "```\n");
  parser_->processToken(taskId, kToolCallEndToken, "");
  parser_->processToken(taskId, kToolCallsEndToken, "");

  auto toolCalls = parser_->finalizeTask(taskId);
  // Parser clears buffer on each sep, so function name should still work
  ASSERT_TRUE(toolCalls.has_value());
  ASSERT_EQ(toolCalls->size(), 1);
}

TEST_F(ToolCallParserTest, StreamingMultipleToolCalls) {
  uint32_t taskId = 100;

  parser_->initializeTask(taskId);

  // First tool call: get_weather
  parser_->processToken(taskId, kToolCallsBeginToken, "");
  parser_->processToken(taskId, kToolCallBeginToken, "");
  parser_->processToken(taskId, 12345, "function");
  parser_->processToken(taskId, kToolSepToken, "");
  parser_->processToken(taskId, 12346, "get_weather\n");
  parser_->processToken(taskId, 12347, "```json\n");
  parser_->processToken(taskId, 12348, "{\"location\":\"SF\"}\n");
  parser_->processToken(taskId, 12349, "```\n");
  parser_->processToken(taskId, kToolCallEndToken, "");

  // Second tool call: get_time
  parser_->processToken(taskId, kToolCallBeginToken, "");
  parser_->processToken(taskId, 12350, "function");
  parser_->processToken(taskId, kToolSepToken, "");
  parser_->processToken(taskId, 12351, "get_time\n");
  parser_->processToken(taskId, 12352, "```json\n");
  parser_->processToken(taskId, 12353, "{\"timezone\":\"PST\"}\n");
  parser_->processToken(taskId, 12354, "```\n");
  parser_->processToken(taskId, kToolCallEndToken, "");

  parser_->processToken(taskId, kToolCallsEndToken, "");

  // Finalize and check
  auto toolCalls = parser_->finalizeTask(taskId);
  ASSERT_TRUE(toolCalls.has_value());
  ASSERT_EQ(toolCalls->size(), 2);

  auto& toolCall1 = (*toolCalls)[0];
  EXPECT_EQ(toolCall1["function"]["name"].asString(), "get_weather");

  auto& toolCall2 = (*toolCalls)[1];
  EXPECT_EQ(toolCall2["function"]["name"].asString(), "get_time");
}

}  // namespace
