// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <cassert>
#include <iostream>
#include <string>

#include "services/reasoning_parser.hpp"

using namespace tt::services;

void testParseComplete() {
  std::cout << "\n=== Testing parseComplete ===\n";

  ReasoningParser parser;

  // Test 1: Complete reasoning block
  {
    std::string input =
        "<think>\nOkay, the user is asking about math.\n</think>\nThe answer "
        "is 42.";
    auto result = parser.parseComplete(input);

    assert(result.has_reasoning);
    assert(!result.is_malformed);
    assert(result.reasoning.has_value());
    assert(result.reasoning.value() == "Okay, the user is asking about math.");
    assert(result.answer == "The answer is 42.");
    std::cout << "✓ Test 1 passed: Complete reasoning block\n";
  }

  // Test 2: No reasoning block
  {
    std::string input = "The answer is 42.";
    auto result = parser.parseComplete(input);

    assert(!result.has_reasoning);
    assert(!result.is_malformed);
    assert(!result.reasoning.has_value());
    assert(result.answer == "The answer is 42.");
    std::cout << "✓ Test 2 passed: No reasoning block\n";
  }

  // Test 3: Malformed (missing </think>)
  {
    std::string input = "<think>\nOkay, the user is asking about math.";
    auto result = parser.parseComplete(input);

    assert(result.has_reasoning);
    assert(result.is_malformed);
    assert(result.reasoning.has_value());
    assert(result.reasoning.value() == "Okay, the user is asking about math.");
    assert(result.answer == "");
    std::cout << "✓ Test 3 passed: Malformed (missing </think>)\n";
  }

  // Test 4: Empty reasoning
  {
    std::string input = "<think>\n\n</think>\nThe answer is 42.";
    auto result = parser.parseComplete(input);

    assert(result.has_reasoning);
    assert(!result.is_malformed);
    // Empty reasoning after trimming should be empty string
    assert(result.answer == "The answer is 42.");
    std::cout << "✓ Test 4 passed: Empty reasoning\n";
  }

  // Test 5: Multi-line reasoning
  {
    std::string input =
        "<think>\nFirst, I need to understand the question.\nThen I'll "
        "calculate.\nFinally, I'll provide the answer.\n</think>\nThe answer "
        "is 42.";
    auto result = parser.parseComplete(input);

    assert(result.has_reasoning);
    assert(!result.is_malformed);
    assert(result.reasoning.has_value());
    assert(result.reasoning.value().find("First") != std::string::npos);
    assert(result.reasoning.value().find("Finally") != std::string::npos);
    assert(result.answer == "The answer is 42.");
    std::cout << "✓ Test 5 passed: Multi-line reasoning\n";
  }

  std::cout << "✅ All parseComplete tests passed!\n";
}

void testStreamingTokens() {
  std::cout << "\n=== Testing Streaming Tokens ===\n";

  ReasoningParser parser;
  std::string taskId = "test-task-123";

  // Initialize task
  parser.initializeTask(taskId);
  assert(parser.activeTaskCount() == 1);
  std::cout << "✓ Task initialized\n";

  // Simulate token sequence: <think> reasoning </think> answer
  std::vector<int64_t> tokens = {
      ReasoningParser::THINK_START_TOKEN,  // <think>
      201,                                 // \n
      12345,                               // "reasoning"
      67890,                               // "text"
      ReasoningParser::THINK_END_TOKEN,    // </think>
      201,                                 // \n
      11111,                               // "answer"
      22222,                               // "text"
      1                                    // EOS
  };

  std::vector<bool> expected = {
      false,  // <think> - filtered
      false,  // \n - filtered (in reasoning)
      false,  // reasoning - filtered
      false,  // text - filtered
      false,  // </think> - filtered
      true,   // \n - emitted
      true,   // answer - emitted
      true,   // text - emitted
      true    // EOS - emitted
  };

  for (size_t i = 0; i < tokens.size(); ++i) {
    bool shouldEmit = parser.processToken(taskId, tokens[i]);
    assert(shouldEmit == expected[i]);
  }

  std::cout << "✓ Token filtering correct\n";

  // Check state
  assert(!parser.isInReasoning(taskId));
  std::cout << "✓ Final state: not in reasoning\n";

  // Finalize
  parser.finalizeTask(taskId);
  assert(parser.activeTaskCount() == 0);
  std::cout << "✓ Task finalized\n";

  std::cout << "✅ All streaming token tests passed!\n";
}

void testMultipleTasks() {
  std::cout << "\n=== Testing Multiple Concurrent Tasks ===\n";

  ReasoningParser parser;

  // Initialize multiple tasks
  for (int i = 0; i < 512; ++i) {
    std::string taskId = "task-" + std::to_string(i);
    parser.initializeTask(taskId);
  }

  assert(parser.activeTaskCount() == 512);
  std::cout << "✓ Initialized 512 tasks\n";

  // Process tokens for different tasks in interleaved manner
  for (int i = 0; i < 512; i += 2) {
    std::string taskId = "task-" + std::to_string(i);

    // Task enters reasoning
    bool shouldEmit =
        parser.processToken(taskId, ReasoningParser::THINK_START_TOKEN);
    assert(!shouldEmit);
    assert(parser.isInReasoning(taskId));
  }

  std::cout << "✓ Even-numbered tasks in reasoning\n";

  // Check odd tasks are not in reasoning
  for (int i = 1; i < 512; i += 2) {
    std::string taskId = "task-" + std::to_string(i);
    assert(!parser.isInReasoning(taskId));
  }

  std::cout << "✓ Odd-numbered tasks not in reasoning\n";

  // Exit reasoning for even tasks
  for (int i = 0; i < 512; i += 2) {
    std::string taskId = "task-" + std::to_string(i);
    bool shouldEmit =
        parser.processToken(taskId, ReasoningParser::THINK_END_TOKEN);
    assert(!shouldEmit);  // </think> itself is filtered
    assert(!parser.isInReasoning(taskId));
  }

  std::cout << "✓ Even-numbered tasks exited reasoning\n";

  // Finalize all tasks
  for (int i = 0; i < 512; ++i) {
    std::string taskId = "task-" + std::to_string(i);
    parser.finalizeTask(taskId);
  }

  assert(parser.activeTaskCount() == 0);
  std::cout << "✓ All tasks finalized\n";

  std::cout << "✅ All multi-task tests passed!\n";
}

void testEdgeCases() {
  std::cout << "\n=== Testing Edge Cases ===\n";

  ReasoningParser parser;

  // Test 1: Uninitialized task
  {
    bool shouldEmit = parser.processToken("uninitialized", 12345);
    assert(shouldEmit);  // Should emit by default
    std::cout << "✓ Test 1 passed: Uninitialized task emits by default\n";
  }

  // Test 2: </think> without <think>
  {
    std::string taskId = "malformed-task";
    parser.initializeTask(taskId);

    // Process </think> without <think>
    bool shouldEmit =
        parser.processToken(taskId, ReasoningParser::THINK_END_TOKEN);
    assert(!shouldEmit);  // Tag itself is filtered
    assert(!parser.isInReasoning(taskId));

    parser.finalizeTask(taskId);
    std::cout << "✓ Test 2 passed: </think> without <think> handled\n";
  }

  // Test 3: Multiple <think> tags
  {
    std::string taskId = "multi-think";
    parser.initializeTask(taskId);

    parser.processToken(taskId, ReasoningParser::THINK_START_TOKEN);
    assert(parser.isInReasoning(taskId));

    // Another <think> while already in reasoning
    bool shouldEmit =
        parser.processToken(taskId, ReasoningParser::THINK_START_TOKEN);
    assert(!shouldEmit);  // Still filtered
    assert(parser.isInReasoning(taskId));

    parser.processToken(taskId, ReasoningParser::THINK_END_TOKEN);
    assert(!parser.isInReasoning(taskId));

    parser.finalizeTask(taskId);
    std::cout << "✓ Test 3 passed: Multiple <think> tags handled\n";
  }

  // Test 4: Finalize while in reasoning
  {
    std::string taskId = "incomplete";
    parser.initializeTask(taskId);

    parser.processToken(taskId, ReasoningParser::THINK_START_TOKEN);
    assert(parser.isInReasoning(taskId));

    // Finalize without closing tag (logs warning but doesn't crash)
    parser.finalizeTask(taskId);
    assert(parser.activeTaskCount() == 0);

    std::cout << "✓ Test 4 passed: Finalize while in reasoning handled\n";
  }

  std::cout << "✅ All edge case tests passed!\n";
}

int main() {
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║     DeepSeek R1 Reasoning Parser Test Suite             ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";

  try {
    testParseComplete();
    testStreamingTokens();
    testMultipleTasks();
    testEdgeCases();

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
