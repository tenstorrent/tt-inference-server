// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <cassert>
#include <iostream>
#include <string>

#include "config/types.hpp"
#include "services/tool_call_parser.hpp"

using namespace tt::services;

void testStreamingTokens() {
  std::cout << "\n=== Testing Streaming Tokens ===\n";

  auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);
  uint32_t taskId = 1;

  parser->initializeTask(taskId);
  assert(parser->activeTaskCount() == 1);
  std::cout << "✓ Task initialized\n";

  // Token IDs (from DeepSeek tokenizer)
  constexpr int64_t TOOL_CALLS_BEGIN_TOKEN = 128806;
  constexpr int64_t TOOL_CALLS_END_TOKEN = 128807;
  constexpr int64_t TOOL_CALL_BEGIN_TOKEN = 128808;
  constexpr int64_t TOOL_CALL_END_TOKEN = 128809;
  constexpr int64_t TOOL_SEP_TOKEN = 128814;

  // Simulate: <｜tool▁calls▁begin｜>
  {
    auto r = parser->processToken(taskId, TOOL_CALLS_BEGIN_TOKEN, "");
    assert(!r.has_value());                // No delta to emit
    assert(parser->isInToolCall(taskId));  // But we're in tool call mode
  }

  // Simulate: <｜tool▁call▁begin｜>
  {
    auto r = parser->processToken(taskId, TOOL_CALL_BEGIN_TOKEN, "");
    assert(!r.has_value());
  }

  // Simulate: "function"
  {
    auto r = parser->processToken(taskId, 12345, "function");
    assert(!r.has_value());
  }

  // Simulate: <｜tool▁sep｜>
  {
    auto r = parser->processToken(taskId, TOOL_SEP_TOKEN, "");
    assert(!r.has_value());
  }

  // Simulate: "get_weather\n" - this emits TOOL_CALL_START
  {
    auto r = parser->processToken(taskId, 12346, "get_weather\n");
    assert(r.has_value());
    assert(r->delta_type == ToolCallDeltaType::TOOL_CALL_START);
    assert(r->function_name == "get_weather");
  }

  // Simulate: "```json\n"
  {
    auto r = parser->processToken(taskId, 12347, "```json\n");
    assert(!r.has_value());
  }

  // Simulate: JSON arguments - emits ARGUMENTS_DELTA
  {
    auto r = parser->processToken(taskId, 12348,
                                  "{\"location\":\"San Francisco\"}\n");
    assert(r.has_value());
    assert(r->delta_type == ToolCallDeltaType::ARGUMENTS_DELTA);
  }

  // Simulate: "```\n"
  {
    auto r = parser->processToken(taskId, 12349, "```\n");
    assert(!r.has_value());
  }

  // Simulate: <｜tool▁call▁end｜> - emits TOOL_CALL_END
  {
    auto r = parser->processToken(taskId, TOOL_CALL_END_TOKEN, "");
    assert(r.has_value());
    assert(r->delta_type == ToolCallDeltaType::TOOL_CALL_END);
  }

  // Simulate: <｜tool▁calls▁end｜>
  {
    auto r = parser->processToken(taskId, TOOL_CALLS_END_TOKEN, "");
    assert(!r.has_value());
    assert(!parser->isInToolCall(taskId));  // Exited tool call mode
  }

  // Regular text after tool calls - parser returns nullopt, not in tool call
  {
    auto r = parser->processToken(taskId, 11111, "The answer is ready.");
    assert(!r.has_value());
    assert(!parser->isInToolCall(taskId));  // Caller handles as regular text
  }

  // Finalize and check tool calls were parsed
  auto toolCalls = parser->finalizeTask(taskId);
  assert(toolCalls.has_value());
  assert(toolCalls->isArray());
  assert(toolCalls->size() == 1);

  auto& toolCall = (*toolCalls)[0];
  assert(toolCall["id"].asString() == "call_0");
  assert(toolCall["type"].asString() == "function");
  assert(toolCall["function"]["name"].asString() == "get_weather");

  assert(parser->activeTaskCount() == 0);
  std::cout << "✓ Token classification correct\n";
  std::cout << "✓ Tool call parsed from streaming tokens\n";
  std::cout << "✓ Task finalized\n";

  std::cout << "✅ All streaming token tests passed!\n";
}

void testMultipleStreamingTasks() {
  std::cout << "\n=== Testing Multiple Concurrent Streaming Tasks ===\n";

  auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);

  // Initialize multiple tasks
  for (uint32_t i = 0; i < 10; ++i) {
    parser->initializeTask(i);
  }

  assert(parser->activeTaskCount() == 10);
  std::cout << "✓ Initialized 10 tasks\n";

  constexpr int64_t TOOL_CALLS_BEGIN_TOKEN = 128806;

  // Process tokens for different tasks in interleaved manner
  for (uint32_t i = 0; i < 10; i += 2) {
    auto r = parser->processToken(i, TOOL_CALLS_BEGIN_TOKEN, "");
    assert(!r.has_value());           // No delta to emit
    assert(parser->isInToolCall(i));  // But in tool call mode
  }

  std::cout << "✓ Even-numbered tasks in tool call mode\n";

  // Check odd tasks are not in tool call mode
  for (uint32_t i = 1; i < 10; i += 2) {
    assert(!parser->isInToolCall(i));
  }

  std::cout << "✓ Odd-numbered tasks not in tool call mode\n";

  // Finalize all tasks
  for (uint32_t i = 0; i < 10; ++i) {
    parser->finalizeTask(i);
  }

  assert(parser->activeTaskCount() == 0);
  std::cout << "✓ All tasks finalized\n";

  std::cout << "✅ All multi-task streaming tests passed!\n";
}

void testStreamingEdgeCases() {
  std::cout << "\n=== Testing Streaming Edge Cases ===\n";

  auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);

  constexpr int64_t TOOL_CALLS_BEGIN_TOKEN = 128806;
  constexpr int64_t TOOL_CALLS_END_TOKEN = 128807;

  // Test 1: Uninitialized task - returns nullopt, caller handles as regular
  {
    auto r = parser->processToken(99999, 12345, "text");
    assert(!r.has_value());
    assert(!parser->isInToolCall(99999));  // Not in tool call mode
    std::cout << "✓ Test 1 passed: Uninitialized task returns nullopt\n";
  }

  // Test 2: Finalize while in tool call
  {
    uint32_t taskId = 50;
    parser->initializeTask(taskId);

    parser->processToken(taskId, TOOL_CALLS_BEGIN_TOKEN, "");
    assert(parser->isInToolCall(taskId));

    auto result = parser->finalizeTask(taskId);
    assert(parser->activeTaskCount() == 0);

    std::cout << "✓ Test 2 passed: Finalize while in tool call handled\n";
  }

  // Test 3: Regular content before and after tool calls
  {
    uint32_t taskId = 51;
    parser->initializeTask(taskId);

    // Regular text before - returns nullopt, not in tool call
    {
      auto r = parser->processToken(taskId, 12345, "Let me help you.");
      assert(!r.has_value());
      assert(!parser->isInToolCall(taskId));  // Caller handles as regular
    }

    // Tool calls block
    parser->processToken(taskId, TOOL_CALLS_BEGIN_TOKEN, "");
    assert(parser->isInToolCall(taskId));

    parser->processToken(taskId, TOOL_CALLS_END_TOKEN, "");
    assert(!parser->isInToolCall(taskId));

    // Regular text after - returns nullopt, not in tool call
    {
      auto r = parser->processToken(taskId, 12346, "Done.");
      assert(!r.has_value());
      assert(!parser->isInToolCall(taskId));
    }

    parser->finalizeTask(taskId);
    std::cout << "✓ Test 3 passed: Regular content before/after tool calls\n";
  }

  std::cout << "✅ All streaming edge case tests passed!\n";
}

void testStreamingMultipleToolCalls() {
  std::cout << "\n=== Testing Streaming Multiple Tool Calls ===\n";

  auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);
  uint32_t taskId = 100;

  parser->initializeTask(taskId);

  constexpr int64_t TOOL_CALLS_BEGIN_TOKEN = 128806;
  constexpr int64_t TOOL_CALLS_END_TOKEN = 128807;
  constexpr int64_t TOOL_CALL_BEGIN_TOKEN = 128808;
  constexpr int64_t TOOL_CALL_END_TOKEN = 128809;
  constexpr int64_t TOOL_SEP_TOKEN = 128814;

  // First tool call: get_weather
  parser->processToken(taskId, TOOL_CALLS_BEGIN_TOKEN, "");
  parser->processToken(taskId, TOOL_CALL_BEGIN_TOKEN, "");
  parser->processToken(taskId, 12345, "function");
  parser->processToken(taskId, TOOL_SEP_TOKEN, "");
  parser->processToken(taskId, 12346, "get_weather\n");
  parser->processToken(taskId, 12347, "```json\n");
  parser->processToken(taskId, 12348, "{\"location\":\"SF\"}\n");
  parser->processToken(taskId, 12349, "```\n");
  parser->processToken(taskId, TOOL_CALL_END_TOKEN, "");

  // Second tool call: get_time
  parser->processToken(taskId, TOOL_CALL_BEGIN_TOKEN, "");
  parser->processToken(taskId, 12350, "function");
  parser->processToken(taskId, TOOL_SEP_TOKEN, "");
  parser->processToken(taskId, 12351, "get_time\n");
  parser->processToken(taskId, 12352, "```json\n");
  parser->processToken(taskId, 12353, "{\"timezone\":\"PST\"}\n");
  parser->processToken(taskId, 12354, "```\n");
  parser->processToken(taskId, TOOL_CALL_END_TOKEN, "");

  parser->processToken(taskId, TOOL_CALLS_END_TOKEN, "");

  // Finalize and check
  auto toolCalls = parser->finalizeTask(taskId);
  assert(toolCalls.has_value());
  assert(toolCalls->size() == 2);

  auto& toolCall1 = (*toolCalls)[0];
  assert(toolCall1["function"]["name"].asString() == "get_weather");

  auto& toolCall2 = (*toolCalls)[1];
  assert(toolCall2["function"]["name"].asString() == "get_time");

  std::cout << "✓ Multiple tool calls parsed correctly\n";
  std::cout << "✅ All streaming multiple tool call tests passed!\n";
}

int main() {
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║         Tool Call Parser Test Suite                     ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";

  try {
    testStreamingTokens();
    testMultipleStreamingTasks();
    testStreamingEdgeCases();
    testStreamingMultipleToolCalls();

    std::cout << "\n";
    std::cout
        << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout
        << "║              🎉 ALL TESTS PASSED! 🎉                    ║\n";
    std::cout
        << "╚══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "\n❌ TEST FAILED: " << e.what() << "\n";
    return 1;
  }
}
