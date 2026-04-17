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

  // Test 2: Multiple tool calls
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

    std::cout << "✓ Test 2 passed: Multiple tool calls\n";
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
    assert(result.find("<｜tool▁calls▁begin｜>") == std::string::npos);
    assert(result.find("<｜tool▁calls▁end｜>") == std::string::npos);
    assert(result.find("Some text before") != std::string::npos ||
           result.find("Some text after") != std::string::npos ||
           result.empty());

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
