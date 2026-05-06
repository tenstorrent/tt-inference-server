// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <cassert>
#include <iostream>
#include <string>

#include "config/types.hpp"
#include "services/tool_call_parser.hpp"

using namespace tt::services;

void testDeepSeekToolCallParsing() {
  std::cout << "\n=== Testing DeepSeek Tool Call Parsing ===\n";

  auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);

  // Test 1: Single tool call
  {
    std::string input =
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_"
        "weather\n"
        "```json\n"
        "{\"location\":\"San Francisco\"}\n"
        "```\n"
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>";

    auto result = parser->parseComplete(input);
    assert(result.has_value());
    assert(result->isArray());
    assert(result->size() == 1);

    auto& toolCall = (*result)[0];
    assert(toolCall["id"].asString() == "call_0");
    assert(toolCall["type"].asString() == "function");
    assert(toolCall["function"]["name"].asString() == "get_weather");

    std::string args = toolCall["function"]["arguments"].asString();
    assert(args.find("San Francisco") != std::string::npos);

    std::cout << "✓ Test 1 passed: Single tool call\n";
  }

  // Test 2: Multiple tool calls, with parallelToolCalls set to true
  {
    std::string input =
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        "```json\n"
        "{\"location\":\"San Francisco\"}\n"
        "```\n"
        "<｜tool▁call▁end｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n"
        "```json\n"
        "{\"timezone\":\"America/Los_Angeles\"}\n"
        "```\n"
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>";

    auto result = parser->parseComplete(input);
    assert(result.has_value());
    assert(result->isArray());
    assert(result->size() == 2);

    // First tool call
    auto& toolCall1 = (*result)[0];
    assert(toolCall1["id"].asString() == "call_0");
    assert(toolCall1["function"]["name"].asString() == "get_weather");

    // Second tool call
    auto& toolCall2 = (*result)[1];
    assert(toolCall2["id"].asString() == "call_1");
    assert(toolCall2["function"]["name"].asString() == "get_time");

    std::cout << "✓ Test 2 passed: Multiple tool calls with parallelToolCalls "
                 "set to true\n";
  }

  // Test 3: No tool calls
  {
    std::string input = "Just a regular response without any tool calls.";
    auto result = parser->parseComplete(input);
    assert(!result.has_value());
    std::cout << "✓ Test 3 passed: No tool calls\n";
  }

  // Test 4: Complex JSON arguments with objects and arrays
  {
    std::string input =
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
        "create_event\n"
        "```json\n"
        "{\"title\":\"Team Meeting\",\"location\":{\"venue\":\"Conference Room "
        "A\",\"address\":\"123 Main "
        "St\"},\"attendees\":[{\"name\":\"Alice\",\"email\":\"alice@example."
        "com\"},{\"name\":\"Bob\",\"email\":\"bob@example.com\"}],\"start_"
        "time\":\"2026-04-20T10:00:00Z\",\"duration_minutes\":60}\n"
        "```\n"
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>";

    auto result = parser->parseComplete(input);
    assert(result.has_value());
    assert(result->size() == 1);

    auto& toolCall = (*result)[0];
    assert(toolCall["function"]["name"].asString() == "create_event");

    std::string args = toolCall["function"]["arguments"].asString();
    assert(args.find("Team Meeting") != std::string::npos);
    assert(args.find("Conference Room A") != std::string::npos);
    assert(args.find("Alice") != std::string::npos);
    assert(args.find("alice@example.com") != std::string::npos);
    assert(args.find("attendees") != std::string::npos);

    std::cout
        << "✓ Test 4 passed: Complex JSON arguments with objects and arrays\n";
  }

  // Test 5: Tool call with text before/after
  {
    std::string input =
        "Let me check that for you.\n"
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_"
        "weather\n"
        "```json\n"
        "{\"location\":\"San Francisco\"}\n"
        "```\n"
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n"
        "I'll get that information now.";

    auto result = parser->parseComplete(input);
    assert(result.has_value());
    assert(result->size() == 1);

    auto& toolCall = (*result)[0];
    assert(toolCall["id"].asString() == "call_0");
    assert(toolCall["type"].asString() == "function");
    assert(toolCall["function"]["name"].asString() == "get_weather");

    std::string args = toolCall["function"]["arguments"].asString();
    assert(args.find("San Francisco") != std::string::npos);

    std::cout << "✓ Test 5 passed: Tool call with surrounding text\n";
  }

  // Test 6: Multiple tool calls, with parallelToolCalls set to false
  {
    std::string input =
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        "```json\n"
        "{\"location\":\"San Francisco\"}\n"
        "```\n"
        "<｜tool▁call▁end｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n"
        "```json\n"
        "{\"timezone\":\"America/Los_Angeles\"}\n"
        "```\n"
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>";

    auto result = parser->parseComplete(input, false);
    assert(result.has_value());
    assert(result->isArray());
    assert(result->size() == 1);

    // First tool call
    auto& toolCall1 = (*result)[0];
    assert(toolCall1["id"].asString() == "call_0");
    assert(toolCall1["function"]["name"].asString() == "get_weather");

    std::cout << "✓ Test 6 passed: Multiple tool calls with parallelToolCalls "
                 "set to false\n";
  }

  std::cout << "✅ All DeepSeek tool call parsing tests passed!\n";
}

void testDeepSeekStripMarkers() {
  std::cout << "\n=== Testing Strip Markers ===\n";

  auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);

  // Test 1: Strip tool call markers
  {
    std::string input =
        "Some text before\n"
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_"
        "weather\n"
        "```json\n"
        "{\"location\":\"San Francisco\"}\n"
        "```\n"
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n"
        "Some text after";

    std::string result = parser->stripMarkers(input);
    assert(result == "Some text before\nSome text after");

    std::cout << "✓ Test 1 passed: Strip tool call markers\n";
  }

  // Test 2: No markers to strip
  {
    std::string input = "Just regular text";
    std::string result = parser->stripMarkers(input);
    assert(result == input);
    std::cout << "✓ Test 2 passed: No markers to strip\n";
  }

  std::cout << "✅ All strip markers tests passed!\n";
}

void testDeepSeekEdgeCases() {
  std::cout << "\n=== Testing Edge Cases ===\n";

  auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);

  // Test 1: Malformed - missing end marker
  {
    std::string input =
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_"
        "weather\n"
        "```json\n"
        "{\"location\":\"San Francisco\"}\n"
        "```\n";

    auto result = parser->parseComplete(input);
    // Should handle gracefully - either return empty or partial parse
    std::cout << "✓ Test 1 passed: Malformed input handled\n";
  }

  // Test 2: Empty function name
  {
    std::string input =
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>\n"
        "```json\n"
        "{\"location\":\"San Francisco\"}\n"
        "```\n"
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>";

    auto result = parser->parseComplete(input);
    // Should handle gracefully
    std::cout << "✓ Test 2 passed: Empty function name handled\n";
  }

  // Test 3: Invalid JSON
  {
    std::string input =
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_"
        "weather\n"
        "```json\n"
        "{invalid json here}\n"
        "```\n"
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>";

    auto result = parser->parseComplete(input);
    // Should handle gracefully - either skip this call or return partial
    std::cout << "✓ Test 3 passed: Invalid JSON handled\n";
  }

  std::cout << "✅ All edge case tests passed!\n";
}

void testDeepSeekOpenAIFormatCompliance() {
  std::cout << "\n=== Testing OpenAI Format Compliance ===\n";

  auto parser = createToolCallParser(tt::config::ModelType::DEEPSEEK_R1_0528);

  std::string input =
      "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_"
      "weather\n"
      "```json\n"
      "{\"location\":\"San Francisco\"}\n"
      "```\n"
      "<｜tool▁call▁end｜><｜tool▁calls▁end｜>";

  auto result = parser->parseComplete(input);
  assert(result.has_value());
  assert(result->isArray());

  auto& toolCall = (*result)[0];

  // Verify OpenAI format structure
  assert(toolCall.isMember("id"));
  assert(toolCall.isMember("type"));
  assert(toolCall.isMember("function"));

  assert(toolCall["type"].asString() == "function");

  auto& function = toolCall["function"];
  assert(function.isMember("name"));
  assert(function.isMember("arguments"));

  // Arguments should be a JSON string, not a parsed object
  assert(function["arguments"].isString());

  std::cout << "✓ OpenAI format compliance verified\n";
  std::cout << "  - id field present\n";
  std::cout << "  - type is 'function'\n";
  std::cout << "  - function.name present\n";
  std::cout << "  - function.arguments is string (not object)\n";

  std::cout << "✅ All OpenAI format compliance tests passed!\n";
}

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
    assert(!r.should_emit);
    assert(r.type == ToolCallContentType::TOOL_CALL);
    assert(parser->isInToolCall(taskId));
  }

  // Simulate: <｜tool▁call▁begin｜>
  {
    auto r = parser->processToken(taskId, TOOL_CALL_BEGIN_TOKEN, "");
    assert(!r.should_emit);
    assert(r.type == ToolCallContentType::TOOL_CALL);
  }

  // Simulate: "function"
  {
    auto r = parser->processToken(taskId, 12345, "function");
    assert(!r.should_emit);
    assert(r.type == ToolCallContentType::TOOL_CALL);
  }

  // Simulate: <｜tool▁sep｜>
  {
    auto r = parser->processToken(taskId, TOOL_SEP_TOKEN, "");
    assert(!r.should_emit);
    assert(r.type == ToolCallContentType::TOOL_CALL);
  }

  // Simulate: "get_weather\n"
  {
    auto r = parser->processToken(taskId, 12346, "get_weather\n");
    assert(!r.should_emit);
    assert(r.type == ToolCallContentType::TOOL_CALL);
  }

  // Simulate: "```json\n"
  {
    auto r = parser->processToken(taskId, 12347, "```json\n");
    assert(!r.should_emit);
    assert(r.type == ToolCallContentType::TOOL_CALL);
  }

  // Simulate: JSON arguments
  {
    auto r = parser->processToken(taskId, 12348, "{\"location\":\"San Francisco\"}\n");
    assert(!r.should_emit);
    assert(r.type == ToolCallContentType::TOOL_CALL);
  }

  // Simulate: "```\n"
  {
    auto r = parser->processToken(taskId, 12349, "```\n");
    assert(!r.should_emit);
    assert(r.type == ToolCallContentType::TOOL_CALL);
  }

  // Simulate: <｜tool▁call▁end｜>
  {
    auto r = parser->processToken(taskId, TOOL_CALL_END_TOKEN, "");
    assert(!r.should_emit);
    assert(r.type == ToolCallContentType::TOOL_CALL);
  }

  // Simulate: <｜tool▁calls▁end｜>
  {
    auto r = parser->processToken(taskId, TOOL_CALLS_END_TOKEN, "");
    assert(!r.should_emit);
    assert(r.type == ToolCallContentType::TOOL_CALL);
    assert(!parser->isInToolCall(taskId));
  }

  // Regular text after tool calls
  {
    auto r = parser->processToken(taskId, 11111, "The answer is ready.");
    assert(r.should_emit);
    assert(r.type == ToolCallContentType::REGULAR);
    assert(r.text == "The answer is ready.");
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
    assert(!r.should_emit);
    assert(parser->isInToolCall(i));
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

  // Test 1: Uninitialized task
  {
    auto r = parser->processToken(99999, 12345, "text");
    assert(r.should_emit);
    assert(r.type == ToolCallContentType::REGULAR);
    std::cout << "✓ Test 1 passed: Uninitialized task emits as regular\n";
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

    // Regular text before
    {
      auto r = parser->processToken(taskId, 12345, "Let me help you.");
      assert(r.should_emit);
      assert(r.type == ToolCallContentType::REGULAR);
    }

    // Tool calls block
    parser->processToken(taskId, TOOL_CALLS_BEGIN_TOKEN, "");
    assert(parser->isInToolCall(taskId));

    parser->processToken(taskId, TOOL_CALLS_END_TOKEN, "");
    assert(!parser->isInToolCall(taskId));

    // Regular text after
    {
      auto r = parser->processToken(taskId, 12346, "Done.");
      assert(r.should_emit);
      assert(r.type == ToolCallContentType::REGULAR);
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
    testDeepSeekToolCallParsing();
    testDeepSeekStripMarkers();
    testDeepSeekEdgeCases();
    testDeepSeekOpenAIFormatCompliance();
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
