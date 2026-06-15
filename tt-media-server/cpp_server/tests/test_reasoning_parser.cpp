// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <string>
#include <tuple>
#include <vector>

#include "services/reasoning_parser.hpp"

using namespace tt::services;

namespace {

class ReasoningParserTest : public ::testing::Test {
 protected:
  ReasoningParser parser_;
};

TEST_F(ReasoningParserTest, StreamingTokens) {
  uint32_t taskId = 1;

  parser_.initializeTask(taskId);
  EXPECT_EQ(parser_.activeTaskCount(), 1);

  // <think> marker: state flips, not emitted
  {
    auto r =
        parser_.processToken(taskId, ReasoningParser::THINK_START_TOKEN, "");
    EXPECT_FALSE(r.should_emit);
    EXPECT_EQ(r.type, ContentType::REASONING);
    EXPECT_TRUE(parser_.isInReasoning(taskId));
  }

  // Reasoning tokens: emitted as REASONING
  {
    auto r = parser_.processToken(taskId, 201, "\n");
    EXPECT_TRUE(r.should_emit);
    EXPECT_EQ(r.type, ContentType::REASONING);
    EXPECT_EQ(r.text, "\n");
  }
  {
    auto r = parser_.processToken(taskId, 12345, "reasoning");
    EXPECT_TRUE(r.should_emit);
    EXPECT_EQ(r.type, ContentType::REASONING);
    EXPECT_EQ(r.text, "reasoning");
  }
  {
    auto r = parser_.processToken(taskId, 67890, " text");
    EXPECT_TRUE(r.should_emit);
    EXPECT_EQ(r.type, ContentType::REASONING);
    EXPECT_EQ(r.text, " text");
  }

  // </think> marker: state flips, not emitted
  {
    auto r = parser_.processToken(taskId, ReasoningParser::THINK_END_TOKEN, "");
    EXPECT_FALSE(r.should_emit);
    EXPECT_EQ(r.type, ContentType::REASONING);
    EXPECT_FALSE(parser_.isInReasoning(taskId));
  }

  // Answer tokens: emitted as ANSWER
  {
    auto r = parser_.processToken(taskId, 201, "\n");
    EXPECT_TRUE(r.should_emit);
    EXPECT_EQ(r.type, ContentType::ANSWER);
    EXPECT_EQ(r.text, "\n");
  }
  {
    auto r = parser_.processToken(taskId, 11111, "answer");
    EXPECT_TRUE(r.should_emit);
    EXPECT_EQ(r.type, ContentType::ANSWER);
    EXPECT_EQ(r.text, "answer");
  }
  {
    auto r = parser_.processToken(taskId, 22222, " text");
    EXPECT_TRUE(r.should_emit);
    EXPECT_EQ(r.type, ContentType::ANSWER);
  }
  {
    auto r = parser_.processToken(taskId, 1, "");
    EXPECT_TRUE(r.should_emit);
    EXPECT_EQ(r.type, ContentType::ANSWER);
  }

  EXPECT_FALSE(parser_.isInReasoning(taskId));

  parser_.finalizeTask(taskId);
  EXPECT_EQ(parser_.activeTaskCount(), 0);
}

TEST_F(ReasoningParserTest, MultipleConcurrentTasks) {
  // Initialize multiple tasks
  for (uint32_t i = 0; i < 512; ++i) {
    parser_.initializeTask(i);
  }

  EXPECT_EQ(parser_.activeTaskCount(), 512);

  // Process tokens for different tasks in interleaved manner
  for (uint32_t i = 0; i < 512; i += 2) {
    auto r = parser_.processToken(i, ReasoningParser::THINK_START_TOKEN, "");
    EXPECT_FALSE(r.should_emit);
    EXPECT_TRUE(parser_.isInReasoning(i));
  }

  // Check odd tasks are not in reasoning
  for (uint32_t i = 1; i < 512; i += 2) {
    EXPECT_FALSE(parser_.isInReasoning(i));
  }

  // Exit reasoning for even tasks
  for (uint32_t i = 0; i < 512; i += 2) {
    auto r = parser_.processToken(i, ReasoningParser::THINK_END_TOKEN, "");
    EXPECT_FALSE(r.should_emit);
    EXPECT_FALSE(parser_.isInReasoning(i));
  }

  // Finalize all tasks
  for (uint32_t i = 0; i < 512; ++i) {
    parser_.finalizeTask(i);
  }

  EXPECT_EQ(parser_.activeTaskCount(), 0);
}

TEST_F(ReasoningParserTest, UninitializedTaskEmitsAsAnswer) {
  auto r = parser_.processToken(99999, 12345, "text");
  EXPECT_TRUE(r.should_emit);
  EXPECT_EQ(r.type, ContentType::ANSWER);
}

TEST_F(ReasoningParserTest, ThinkEndWithoutThinkStart) {
  uint32_t taskId = 50;
  parser_.initializeTask(taskId);

  auto r = parser_.processToken(taskId, ReasoningParser::THINK_END_TOKEN, "");
  EXPECT_FALSE(r.should_emit);
  EXPECT_FALSE(parser_.isInReasoning(taskId));

  parser_.finalizeTask(taskId);
}

TEST_F(ReasoningParserTest, MultipleThinkStartTags) {
  uint32_t taskId = 51;
  parser_.initializeTask(taskId);

  parser_.processToken(taskId, ReasoningParser::THINK_START_TOKEN, "");
  EXPECT_TRUE(parser_.isInReasoning(taskId));

  auto r = parser_.processToken(taskId, ReasoningParser::THINK_START_TOKEN, "");
  EXPECT_FALSE(r.should_emit);
  EXPECT_TRUE(parser_.isInReasoning(taskId));

  parser_.processToken(taskId, ReasoningParser::THINK_END_TOKEN, "");
  EXPECT_FALSE(parser_.isInReasoning(taskId));

  parser_.finalizeTask(taskId);
}

TEST_F(ReasoningParserTest, FinalizeWhileInReasoning) {
  uint32_t taskId = 52;
  parser_.initializeTask(taskId);

  parser_.processToken(taskId, ReasoningParser::THINK_START_TOKEN, "");
  EXPECT_TRUE(parser_.isInReasoning(taskId));

  parser_.finalizeTask(taskId);
  EXPECT_EQ(parser_.activeTaskCount(), 0);
}

// Helper to simulate the suppression logic from
// LLMService::consumerLoopForWorker
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

class ReasoningSuppressionTest : public ::testing::Test {
 protected:
  ReasoningParser parser_;
};

TEST_F(ReasoningSuppressionTest, SuppressionOnReasoningDroppedAnswerKept) {
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

  auto emitted = simulateConsumerLoop(parser_, 100, true, tokens);
  ASSERT_EQ(emitted.size(), 3);
  EXPECT_EQ(emitted[0].type, ContentType::ANSWER);
  EXPECT_EQ(emitted[0].text, "\n");
  EXPECT_EQ(emitted[1].type, ContentType::ANSWER);
  EXPECT_EQ(emitted[1].text, "answer");
  EXPECT_EQ(emitted[2].type, ContentType::ANSWER);
  EXPECT_EQ(emitted[2].text, " text");
}

TEST_F(ReasoningSuppressionTest, SuppressionOffAllTokensPassThrough) {
  std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
      {ReasoningParser::THINK_START_TOKEN, "", false},
      {12345, "reasoning", false},
      {ReasoningParser::THINK_END_TOKEN, "", false},
      {11111, "answer", true},
  };

  auto emitted = simulateConsumerLoop(parser_, 101, false, tokens);
  ASSERT_EQ(emitted.size(), 2);
  EXPECT_EQ(emitted[0].type, ContentType::REASONING);
  EXPECT_EQ(emitted[0].text, "reasoning");
  EXPECT_EQ(emitted[1].type, ContentType::ANSWER);
  EXPECT_EQ(emitted[1].text, "answer");
}

TEST_F(ReasoningSuppressionTest,
       SuppressionOnFinalReasoningConvertedToEmptyAnswer) {
  std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
      {ReasoningParser::THINK_START_TOKEN, "", false},
      {12345, "only reasoning", true},
  };

  auto emitted = simulateConsumerLoop(parser_, 102, true, tokens);
  ASSERT_EQ(emitted.size(), 1);
  EXPECT_EQ(emitted[0].type, ContentType::ANSWER);
  EXPECT_TRUE(emitted[0].text.empty());
}

TEST_F(ReasoningSuppressionTest, SuppressionOnUnclosedThink) {
  std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
      {ReasoningParser::THINK_START_TOKEN, "", false},
      {12345, "reasoning", false},
      {67890, " continues", false},
      {99999, " more reasoning", true},
  };

  auto emitted = simulateConsumerLoop(parser_, 103, true, tokens);
  ASSERT_EQ(emitted.size(), 1);
  EXPECT_EQ(emitted[0].type, ContentType::ANSWER);
  EXPECT_TRUE(emitted[0].text.empty());
}

TEST_F(ReasoningSuppressionTest, SuppressionOnPureAnswerUnaffected) {
  std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
      {11111, "pure", false},
      {22222, " answer", true},
  };

  auto emitted = simulateConsumerLoop(parser_, 104, true, tokens);
  ASSERT_EQ(emitted.size(), 2);
  EXPECT_EQ(emitted[0].type, ContentType::ANSWER);
  EXPECT_EQ(emitted[0].text, "pure");
  EXPECT_EQ(emitted[1].type, ContentType::ANSWER);
  EXPECT_EQ(emitted[1].text, " answer");
}

TEST_F(ReasoningSuppressionTest, SuppressionOnEmptyThinkBlock) {
  std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
      {ReasoningParser::THINK_START_TOKEN, "", false},
      {201, "\n", false},
      {ReasoningParser::THINK_END_TOKEN, "", false},
      {201, "\n", false},
      {11111, "The answer is 4.", true},
  };

  auto emitted = simulateConsumerLoop(parser_, 105, true, tokens);
  ASSERT_EQ(emitted.size(), 2);
  EXPECT_EQ(emitted[0].type, ContentType::ANSWER);
  EXPECT_EQ(emitted[0].text, "\n");
  EXPECT_EQ(emitted[1].type, ContentType::ANSWER);
  EXPECT_EQ(emitted[1].text, "The answer is 4.");
}

TEST_F(ReasoningSuppressionTest, SuppressionOnStrayThinkEnd) {
  std::vector<std::tuple<int64_t, std::string, bool>> tokens = {
      {11111, "Let me think...", false},
      {ReasoningParser::THINK_END_TOKEN, "", false},
      {22222, " The answer is 4.", true},
  };

  auto emitted = simulateConsumerLoop(parser_, 106, true, tokens);
  ASSERT_EQ(emitted.size(), 2);
  EXPECT_EQ(emitted[0].type, ContentType::ANSWER);
  EXPECT_EQ(emitted[0].text, "Let me think...");
  EXPECT_EQ(emitted[1].type, ContentType::ANSWER);
  EXPECT_EQ(emitted[1].text, " The answer is 4.");
}

}  // namespace
