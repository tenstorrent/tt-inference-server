// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <cassert>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

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
  uint32_t taskId = 1;

  parser.initializeTask(taskId);
  assert(parser.activeTaskCount() == 1);
  std::cout << "✓ Task initialized\n";

  // <think> marker: state flips, not emitted
  {
    auto r =
        parser.processToken(taskId, ReasoningParser::THINK_START_TOKEN, "");
    assert(!r.should_emit);
    assert(r.type == ContentType::REASONING);
    assert(parser.isInReasoning(taskId));
  }

  // Reasoning tokens: emitted as REASONING
  {
    auto r = parser.processToken(taskId, 201, "\n");
    assert(r.should_emit);
    assert(r.type == ContentType::REASONING);
    assert(r.text == "\n");
  }
  {
    auto r = parser.processToken(taskId, 12345, "reasoning");
    assert(r.should_emit);
    assert(r.type == ContentType::REASONING);
    assert(r.text == "reasoning");
  }
  {
    auto r = parser.processToken(taskId, 67890, " text");
    assert(r.should_emit);
    assert(r.type == ContentType::REASONING);
    assert(r.text == " text");
  }

  // </think> marker: state flips, not emitted
  {
    auto r = parser.processToken(taskId, ReasoningParser::THINK_END_TOKEN, "");
    assert(!r.should_emit);
    assert(r.type == ContentType::REASONING);
    assert(!parser.isInReasoning(taskId));
  }

  // Answer tokens: emitted as ANSWER
  {
    auto r = parser.processToken(taskId, 201, "\n");
    assert(r.should_emit);
    assert(r.type == ContentType::ANSWER);
    assert(r.text == "\n");
  }
  {
    auto r = parser.processToken(taskId, 11111, "answer");
    assert(r.should_emit);
    assert(r.type == ContentType::ANSWER);
    assert(r.text == "answer");
  }
  {
    auto r = parser.processToken(taskId, 22222, " text");
    assert(r.should_emit);
    assert(r.type == ContentType::ANSWER);
  }
  {
    auto r = parser.processToken(taskId, 1, "");
    assert(r.should_emit);
    assert(r.type == ContentType::ANSWER);
  }

  assert(!parser.isInReasoning(taskId));
  std::cout << "✓ Token classification correct\n";

  parser.finalizeTask(taskId);
  assert(parser.activeTaskCount() == 0);
  std::cout << "✓ Task finalized\n";

  std::cout << "✅ All streaming token tests passed!\n";
}

void testMultipleTasks() {
  std::cout << "\n=== Testing Multiple Concurrent Tasks ===\n";

  ReasoningParser parser;

  // Initialize multiple tasks
  for (uint32_t i = 0; i < 512; ++i) {
    parser.initializeTask(i);
  }

  assert(parser.activeTaskCount() == 512);
  std::cout << "✓ Initialized 512 tasks\n";

  // Process tokens for different tasks in interleaved manner
  for (uint32_t i = 0; i < 512; i += 2) {
    auto r = parser.processToken(i, ReasoningParser::THINK_START_TOKEN, "");
    assert(!r.should_emit);
    assert(parser.isInReasoning(i));
  }

  std::cout << "✓ Even-numbered tasks in reasoning\n";

  // Check odd tasks are not in reasoning
  for (uint32_t i = 1; i < 512; i += 2) {
    assert(!parser.isInReasoning(i));
  }

  std::cout << "✓ Odd-numbered tasks not in reasoning\n";

  // Exit reasoning for even tasks
  for (uint32_t i = 0; i < 512; i += 2) {
    auto r = parser.processToken(i, ReasoningParser::THINK_END_TOKEN, "");
    assert(!r.should_emit);
    assert(!parser.isInReasoning(i));
  }

  std::cout << "✓ Even-numbered tasks exited reasoning\n";

  // Finalize all tasks
  for (uint32_t i = 0; i < 512; ++i) {
    parser.finalizeTask(i);
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
    auto r = parser.processToken(99999, 12345, "text");
    assert(r.should_emit);
    assert(r.type == ContentType::ANSWER);
    std::cout << "✓ Test 1 passed: Uninitialized task emits as answer\n";
  }

  // Test 2: </think> without <think>
  {
    uint32_t taskId = 50;
    parser.initializeTask(taskId);

    auto r = parser.processToken(taskId, ReasoningParser::THINK_END_TOKEN, "");
    assert(!r.should_emit);
    assert(!parser.isInReasoning(taskId));

    parser.finalizeTask(taskId);
    std::cout << "✓ Test 2 passed: </think> without <think> handled\n";
  }

  // Test 3: Multiple <think> tags
  {
    uint32_t taskId = 51;
    parser.initializeTask(taskId);

    parser.processToken(taskId, ReasoningParser::THINK_START_TOKEN, "");
    assert(parser.isInReasoning(taskId));

    auto r =
        parser.processToken(taskId, ReasoningParser::THINK_START_TOKEN, "");
    assert(!r.should_emit);
    assert(parser.isInReasoning(taskId));

    parser.processToken(taskId, ReasoningParser::THINK_END_TOKEN, "");
    assert(!parser.isInReasoning(taskId));

    parser.finalizeTask(taskId);
    std::cout << "✓ Test 3 passed: Multiple <think> tags handled\n";
  }

  // Test 4: Finalize while in reasoning
  {
    uint32_t taskId = 52;
    parser.initializeTask(taskId);

    parser.processToken(taskId, ReasoningParser::THINK_START_TOKEN, "");
    assert(parser.isInReasoning(taskId));

    parser.finalizeTask(taskId);
    assert(parser.activeTaskCount() == 0);

    std::cout << "✓ Test 4 passed: Finalize while in reasoning handled\n";
  }

  std::cout << "✅ All edge case tests passed!\n";
}

// Simulates the suppression logic from LLMService::consumerLoopForWorker.
// Given a sequence of (token_id, text, isFinal) tuples, returns only the tokens
// that would be emitted to the client when suppress=true/false.
struct EmittedToken {
  ContentType type;
  std::string text;
};

std::vector<EmittedToken> simulateConsumerLoop(
    ReasoningParser& parser, uint32_t taskId, bool suppress,
    const std::vector<std::tuple<int64_t, std::string, bool>>& tokens) {
  std::vector<EmittedToken> emitted;
  parser.initializeTask(taskId);

  for (const auto& [tokenId, text, isFinal] : tokens) {
    TokenParseResult parseResult = parser.processToken(taskId, tokenId, text);

    if (suppress && parseResult.type == ContentType::REASONING) {
      if (isFinal) {
        parseResult = {ContentType::ANSWER, "", true};
      } else {
        continue;
      }
    }

    if ((!parseResult.should_emit || parseResult.text.empty()) && !isFinal) {
      continue;
    }

    emitted.push_back({parseResult.type, parseResult.text});

    if (isFinal) {
      break;
    }
  }

  parser.finalizeTask(taskId);
  return emitted;
}

void testReasoningSuppression() {
  std::cout << "\n=== Testing Reasoning Suppression (enable_reasoning=false) "
               "===\n";

  ReasoningParser parser;

  // Test 1: Suppression ON – reasoning tokens dropped, answer tokens kept
  {
    std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
        {ReasoningParser::THINK_START_TOKEN, "", false},
        {201, "\n", false},
        {12345, "reasoning", false},
        {67890, " text", false},
        {ReasoningParser::THINK_END_TOKEN, "", false},
        {201, "\n", false},
        {11111, "answer", false},
        {22222, " text", true},
    };

    auto emitted = simulateConsumerLoop(parser, 100, true, tokens);
    assert(emitted.size() == 3);
    assert(emitted[0].type == ContentType::ANSWER && emitted[0].text == "\n");
    assert(emitted[1].type == ContentType::ANSWER &&
           emitted[1].text == "answer");
    assert(emitted[2].type == ContentType::ANSWER &&
           emitted[2].text == " text");
    std::cout << "✓ Test 1 passed: Reasoning tokens dropped, answer tokens "
                 "kept\n";
  }

  // Test 2: Suppression OFF – all tokens pass through
  {
    std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
        {ReasoningParser::THINK_START_TOKEN, "", false},
        {12345, "reasoning", false},
        {ReasoningParser::THINK_END_TOKEN, "", false},
        {11111, "answer", true},
    };

    auto emitted = simulateConsumerLoop(parser, 101, false, tokens);
    assert(emitted.size() == 2);
    assert(emitted[0].type == ContentType::REASONING &&
           emitted[0].text == "reasoning");
    assert(emitted[1].type == ContentType::ANSWER &&
           emitted[1].text == "answer");
    std::cout << "✓ Test 2 passed: Without suppression, all tokens pass "
                 "through\n";
  }

  // Test 3: Suppression ON + final token is REASONING – converted to empty
  // ANSWER so the stream terminates properly
  {
    std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
        {ReasoningParser::THINK_START_TOKEN, "", false},
        {12345, "only reasoning", true},
    };

    auto emitted = simulateConsumerLoop(parser, 102, true, tokens);
    assert(emitted.size() == 1);
    assert(emitted[0].type == ContentType::ANSWER && emitted[0].text.empty());
    std::cout << "✓ Test 3 passed: Final REASONING token converted to empty "
                 "ANSWER\n";
  }

  // Test 4: Suppression ON + unclosed <think> – all tokens after <think> are
  // classified as REASONING and dropped
  {
    std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
        {ReasoningParser::THINK_START_TOKEN, "", false},
        {12345, "reasoning", false},
        {67890, " continues", false},
        {99999, " more reasoning", true},
    };

    auto emitted = simulateConsumerLoop(parser, 103, true, tokens);
    assert(emitted.size() == 1);
    assert(emitted[0].type == ContentType::ANSWER && emitted[0].text.empty());
    std::cout << "✓ Test 4 passed: Unclosed <think> – all reasoning tokens "
                 "suppressed\n";
  }

  // Test 5: Suppression ON + no reasoning tokens at all – pure answer passes
  // through unchanged
  {
    std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
        {11111, "pure", false},
        {22222, " answer", true},
    };

    auto emitted = simulateConsumerLoop(parser, 104, true, tokens);
    assert(emitted.size() == 2);
    assert(emitted[0].type == ContentType::ANSWER && emitted[0].text == "pure");
    assert(emitted[1].type == ContentType::ANSWER &&
           emitted[1].text == " answer");
    std::cout
        << "✓ Test 5 passed: Pure answer tokens unaffected by suppression\n";
  }

  // Test 6: Suppression ON + template-injected empty think block (the
  // <think>\n</think>\n pattern from DeepSeek chat template)
  {
    std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
        {ReasoningParser::THINK_START_TOKEN, "", false},
        {201, "\n", false},
        {ReasoningParser::THINK_END_TOKEN, "", false},
        {201, "\n", false},
        {11111, "The answer is 4.", true},
    };

    auto emitted = simulateConsumerLoop(parser, 105, true, tokens);
    assert(emitted.size() == 2);
    assert(emitted[0].type == ContentType::ANSWER && emitted[0].text == "\n");
    assert(emitted[1].type == ContentType::ANSWER &&
           emitted[1].text == "The answer is 4.");
    std::cout << "✓ Test 6 passed: Template-injected empty think block "
                 "suppressed cleanly\n";
  }

  // Test 7: Suppression ON + stray </think> without <think> (observed in
  // DeepSeek-R1-0528 when model "thinks outside of thinking")
  {
    std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
        {11111, "Let me think...", false},
        {ReasoningParser::THINK_END_TOKEN, "", false},
        {22222, " The answer is 4.", true},
    };

    auto emitted = simulateConsumerLoop(parser, 106, true, tokens);
    assert(emitted.size() == 2);
    assert(emitted[0].type == ContentType::ANSWER &&
           emitted[0].text == "Let me think...");
    assert(emitted[1].type == ContentType::ANSWER &&
           emitted[1].text == " The answer is 4.");
    std::cout << "✓ Test 7 passed: Stray </think> without <think> – answer "
                 "tokens pass through\n";
  }

  std::cout << "✅ All reasoning suppression tests passed!\n";
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
    testReasoningSuppression();

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
