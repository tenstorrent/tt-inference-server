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
  constexpr int64_t kToolCallsBeginToken = 128806;
  constexpr int64_t kToolCallsEndToken = 128807;
  constexpr int64_t kToolCallBeginToken = 128808;
  constexpr int64_t kToolCallEndToken = 128809;
  constexpr int64_t kToolSepToken = 128814;

  // Simulate: <｜tool▁calls▁begin｜>
  {
    auto r = parser->processToken(taskId, kToolCallsBeginToken, "");
    assert(!r.has_value());                // No delta to emit
    assert(parser->isInToolCall(taskId));  // But we're in tool call mode
  }

  // Simulate: <｜tool▁call▁begin｜>
  {
    auto r = parser->processToken(taskId, kToolCallBeginToken, "");
    assert(!r.has_value());
  }

  // Simulate: "function"
  {
    auto r = parser->processToken(taskId, 12345, "function");
    assert(!r.has_value());
  }

  // Simulate: <｜tool▁sep｜>
  {
    auto r = parser->processToken(taskId, kToolSepToken, "");
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
    auto r = parser->processToken(taskId, kToolCallEndToken, "");
    assert(r.has_value());
    assert(r->delta_type == ToolCallDeltaType::TOOL_CALL_END);
  }

  // Simulate: <｜tool▁calls▁end｜>
  {
    auto r = parser->processToken(taskId, kToolCallsEndToken, "");
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

  constexpr int64_t kToolCallsBeginToken = 128806;

  // Process tokens for different tasks in interleaved manner
  for (uint32_t i = 0; i < 10; i += 2) {
    auto r = parser->processToken(i, kToolCallsBeginToken, "");
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

  constexpr int64_t kToolCallsBeginToken = 128806;
  constexpr int64_t kToolCallsEndToken = 128807;

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

    parser->processToken(taskId, kToolCallsBeginToken, "");
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
    parser->processToken(taskId, kToolCallsBeginToken, "");
    assert(parser->isInToolCall(taskId));

    parser->processToken(taskId, kToolCallsEndToken, "");
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

void testMalformedSequences() {
  std::cout << "\n=== Testing Malformed Token Sequences ===\n";

  constexpr int64_t kToolCallsBeginToken = 128806;
  constexpr int64_t kToolCallsEndToken = 128807;
  constexpr int64_t kToolCallBeginToken = 128808;
  constexpr int64_t kToolCallEndToken = 128809;
  constexpr int64_t kToolSepToken = 128814;

  // Test 1: <tool_calls_begin> <tool_call_end> <tool_call_begin> ...
  // Missing tool_call_begin before first tool_call_end
  {
    auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);
    uint32_t taskId = 200;
    parser->initializeTask(taskId);

    // Enter tool calls block
    parser->processToken(taskId, kToolCallsBeginToken, "");
    assert(parser->isInToolCall(taskId));

    // Spurious tool_call_end without matching begin - should handle gracefully
    auto r = parser->processToken(taskId, kToolCallEndToken, "");
    // Parser tries to finalize empty tool call (logs warning) and returns end
    // delta

    // Now start a proper tool call
    parser->processToken(taskId, kToolCallBeginToken, "");
    parser->processToken(taskId, 12345, "function");
    parser->processToken(taskId, kToolSepToken, "");
    parser->processToken(taskId, 12346, "get_weather\n");
    parser->processToken(taskId, 12347, "```json\n");
    parser->processToken(taskId, 12348, "{\"location\":\"SF\"}\n");
    parser->processToken(taskId, 12349, "```\n");
    parser->processToken(taskId, kToolCallEndToken, "");
    parser->processToken(taskId, kToolCallsEndToken, "");

    auto toolCalls = parser->finalizeTask(taskId);
    // Should have only the valid tool call, spurious end was ignored
    assert(toolCalls.has_value());
    assert(toolCalls->size() == 1);
    assert((*toolCalls)[0]["function"]["name"].asString() == "get_weather");

    std::cout
        << "✓ Test 1 passed: Spurious tool_call_end before tool_call_begin "
           "handled\n";
  }

  // Test 2: <tool_calls_end> text <tool_calls_begin> ...
  // End token before begin - should treat text as regular content
  {
    auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);
    uint32_t taskId = 201;
    parser->initializeTask(taskId);

    // Spurious end token while in REGULAR state
    auto r1 = parser->processToken(taskId, kToolCallsEndToken, "");
    assert(!r1.has_value());
    assert(!parser->isInToolCall(taskId));  // Still in regular mode

    // Text that should be treated as regular content
    auto r2 = parser->processToken(taskId, 12345, "some tool name");
    assert(!r2.has_value());
    assert(!parser->isInToolCall(taskId));  // Still regular

    // Now proper begin
    parser->processToken(taskId, kToolCallsBeginToken, "");
    assert(parser->isInToolCall(taskId));

    parser->processToken(taskId, kToolCallsEndToken, "");
    assert(!parser->isInToolCall(taskId));

    auto toolCalls = parser->finalizeTask(taskId);
    // No valid tool calls parsed
    assert(!toolCalls.has_value() || toolCalls->size() == 0);

    std::cout << "✓ Test 2 passed: tool_calls_end before begin handled\n";
  }

  // Test 3: Double tool_calls_begin
  {
    auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);
    uint32_t taskId = 202;
    parser->initializeTask(taskId);

    parser->processToken(taskId, kToolCallsBeginToken, "");
    assert(parser->isInToolCall(taskId));

    // Second begin - should stay in tool call mode
    parser->processToken(taskId, kToolCallsBeginToken, "");
    assert(parser->isInToolCall(taskId));

    parser->processToken(taskId, kToolCallsEndToken, "");
    assert(!parser->isInToolCall(taskId));

    parser->finalizeTask(taskId);
    std::cout << "✓ Test 3 passed: Double tool_calls_begin handled\n";
  }

  // Test 4: tool_call_begin without tool_calls_begin (outer wrapper)
  {
    auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);
    uint32_t taskId = 203;
    parser->initializeTask(taskId);

    // Individual tool call markers without outer wrapper
    parser->processToken(taskId, kToolCallBeginToken, "");
    // State machine goes to IN_TOOL_CALL even without outer wrapper
    assert(parser->isInToolCall(taskId));

    parser->processToken(taskId, 12345, "function");
    parser->processToken(taskId, kToolSepToken, "");
    parser->processToken(taskId, 12346, "get_time\n");
    parser->processToken(taskId, 12347, "```json\n");
    parser->processToken(taskId, 12348, "{}\n");
    parser->processToken(taskId, 12349, "```\n");
    parser->processToken(taskId, kToolCallEndToken, "");

    auto toolCalls = parser->finalizeTask(taskId);
    // Should still parse the tool call
    assert(toolCalls.has_value());
    assert(toolCalls->size() == 1);

    std::cout << "✓ Test 4 passed: tool_call without outer wrapper handled\n";
  }

  // Test 5: Incomplete tool call - missing tool_call_end
  {
    auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);
    uint32_t taskId = 204;
    parser->initializeTask(taskId);

    parser->processToken(taskId, kToolCallsBeginToken, "");
    parser->processToken(taskId, kToolCallBeginToken, "");
    parser->processToken(taskId, 12345, "function");
    parser->processToken(taskId, kToolSepToken, "");
    parser->processToken(taskId, 12346, "get_weather\n");
    parser->processToken(taskId, 12347, "```json\n");
    parser->processToken(taskId, 12348, "{\"x\":1}\n");
    // Missing tool_call_end and tool_calls_end - just finalize

    auto toolCalls = parser->finalizeTask(taskId);
    // Incomplete tool call not finalized to array
    assert(!toolCalls.has_value() || toolCalls->size() == 0);

    std::cout << "✓ Test 5 passed: Incomplete tool call handled\n";
  }

  // Test 6: Garbage text between markers
  {
    auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);
    uint32_t taskId = 205;
    parser->initializeTask(taskId);

    parser->processToken(taskId, kToolCallsBeginToken, "");
    // Random garbage text between tool_calls_begin and tool_call_begin
    parser->processToken(taskId, 12345, "random garbage here\n");
    parser->processToken(taskId, 12346, "more nonsense!!!");

    parser->processToken(taskId, kToolCallBeginToken, "");
    parser->processToken(taskId, 12347, "function");
    parser->processToken(taskId, kToolSepToken, "");
    parser->processToken(taskId, 12348, "valid_func\n");
    parser->processToken(taskId, 12349, "```json\n");
    parser->processToken(taskId, 12350, "{}\n");
    parser->processToken(taskId, 12351, "```\n");
    parser->processToken(taskId, kToolCallEndToken, "");
    parser->processToken(taskId, kToolCallsEndToken, "");

    auto toolCalls = parser->finalizeTask(taskId);
    assert(toolCalls.has_value());
    assert(toolCalls->size() == 1);
    assert((*toolCalls)[0]["function"]["name"].asString() == "valid_func");

    std::cout << "✓ Test 6 passed: Garbage text between markers handled\n";
  }

  // Test 7: Multiple consecutive tool_sep tokens
  {
    auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);
    uint32_t taskId = 206;
    parser->initializeTask(taskId);

    parser->processToken(taskId, kToolCallsBeginToken, "");
    parser->processToken(taskId, kToolCallBeginToken, "");
    parser->processToken(taskId, 12345, "function");
    parser->processToken(taskId, kToolSepToken, "");
    parser->processToken(taskId, kToolSepToken, "");  // Extra sep
    parser->processToken(taskId, kToolSepToken, "");  // Another extra
    parser->processToken(taskId, 12346, "my_func\n");
    parser->processToken(taskId, 12347, "```json\n");
    parser->processToken(taskId, 12348, "{\"key\":\"value\"}\n");
    parser->processToken(taskId, 12349, "```\n");
    parser->processToken(taskId, kToolCallEndToken, "");
    parser->processToken(taskId, kToolCallsEndToken, "");

    auto toolCalls = parser->finalizeTask(taskId);
    // Parser clears buffer on each sep, so function name should still work
    assert(toolCalls.has_value());
    assert(toolCalls->size() == 1);

    std::cout << "✓ Test 7 passed: Multiple consecutive tool_sep handled\n";
  }

  std::cout << "✅ All malformed sequence tests passed!\n";
}

void testStreamingMultipleToolCalls() {
  std::cout << "\n=== Testing Streaming Multiple Tool Calls ===\n";

  auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);
  uint32_t taskId = 100;

  parser->initializeTask(taskId);

  constexpr int64_t kToolCallsBeginToken = 128806;
  constexpr int64_t kToolCallsEndToken = 128807;
  constexpr int64_t kToolCallBeginToken = 128808;
  constexpr int64_t kToolCallEndToken = 128809;
  constexpr int64_t kToolSepToken = 128814;

  // First tool call: get_weather
  parser->processToken(taskId, kToolCallsBeginToken, "");
  parser->processToken(taskId, kToolCallBeginToken, "");
  parser->processToken(taskId, 12345, "function");
  parser->processToken(taskId, kToolSepToken, "");
  parser->processToken(taskId, 12346, "get_weather\n");
  parser->processToken(taskId, 12347, "```json\n");
  parser->processToken(taskId, 12348, "{\"location\":\"SF\"}\n");
  parser->processToken(taskId, 12349, "```\n");
  parser->processToken(taskId, kToolCallEndToken, "");

  // Second tool call: get_time
  parser->processToken(taskId, kToolCallBeginToken, "");
  parser->processToken(taskId, 12350, "function");
  parser->processToken(taskId, kToolSepToken, "");
  parser->processToken(taskId, 12351, "get_time\n");
  parser->processToken(taskId, 12352, "```json\n");
  parser->processToken(taskId, 12353, "{\"timezone\":\"PST\"}\n");
  parser->processToken(taskId, 12354, "```\n");
  parser->processToken(taskId, kToolCallEndToken, "");

  parser->processToken(taskId, kToolCallsEndToken, "");

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
    testMalformedSequences();
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
