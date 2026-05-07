// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/llm_response_accumulator.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>

namespace {

using tt::api::LLMResponseAccumulator;
using tt::api::ResponseWriterParams;
using tt::domain::llm::LLMChoice;
using tt::domain::llm::LLMResponse;
using tt::domain::llm::LLMStreamChunk;

constexpr uint32_t TASK_ID = 4242;
constexpr int64_t CREATED_TS = 1700000000;
constexpr int PROMPT_TOKENS = 7;

ResponseWriterParams makeParams(
    std::optional<std::string> sessionId = std::nullopt) {
  ResponseWriterParams params;
  params.completionId = "chatcmpl-test";
  params.model = "test-model";
  params.created = CREATED_TS;
  params.promptTokenCount = PROMPT_TOKENS;
  params.sessionId = std::move(sessionId);
  params.taskId = TASK_ID;
  return params;
}

LLMStreamChunk makeChunkWithText(const std::string& text) {
  LLMStreamChunk chunk(TASK_ID);
  LLMChoice choice;
  choice.text = text;
  chunk.choices.push_back(std::move(choice));
  return chunk;
}

LLMStreamChunk makeChunkWithReasoning(const std::string& reasoning) {
  LLMStreamChunk chunk(TASK_ID);
  LLMChoice choice;
  choice.reasoning = reasoning;
  chunk.choices.push_back(std::move(choice));
  return chunk;
}

LLMStreamChunk makeFinishChunk(const std::string& finishReason) {
  LLMStreamChunk chunk(TASK_ID);
  LLMChoice choice;
  choice.finish_reason = finishReason;
  chunk.choices.push_back(std::move(choice));
  return chunk;
}

LLMStreamChunk makeEmptyChoicesChunk() { return LLMStreamChunk(TASK_ID); }

// ---------------------------------------------------------------------------
// build() with no chunks -- must produce a valid empty response
// ---------------------------------------------------------------------------

TEST(LLMResponseAccumulator, EmptyBuildHasDefaultFinishReasonAndZeroTokens) {
  LLMResponseAccumulator acc;
  const LLMResponse response = acc.build(makeParams());

  EXPECT_EQ(acc.tokenCount(), 0);
  ASSERT_EQ(response.choices.size(), 1u);
  EXPECT_EQ(response.choices[0].text, "");
  EXPECT_FALSE(response.choices[0].reasoning.has_value());
  ASSERT_TRUE(response.choices[0].finish_reason.has_value());
  EXPECT_EQ(response.choices[0].finish_reason.value(), "stop");
  EXPECT_EQ(response.usage.completion_tokens, 0);
  EXPECT_EQ(response.usage.prompt_tokens, PROMPT_TOKENS);
  EXPECT_EQ(response.usage.total_tokens, PROMPT_TOKENS);
}

TEST(LLMResponseAccumulator, BuildPropagatesParamsToResponse) {
  LLMResponseAccumulator acc;
  const LLMResponse response = acc.build(makeParams("sess-abc"));

  EXPECT_EQ(response.task_id, TASK_ID);
  EXPECT_EQ(response.id, "chatcmpl-test");
  EXPECT_EQ(response.model, "test-model");
  EXPECT_EQ(response.created, CREATED_TS);
  ASSERT_TRUE(response.usage.sessionId.has_value());
  EXPECT_EQ(response.usage.sessionId.value(), "sess-abc");
}

TEST(LLMResponseAccumulator, BuildOmitsSessionIdWhenParamsHaveNone) {
  LLMResponseAccumulator acc;
  const LLMResponse response = acc.build(makeParams());
  EXPECT_FALSE(response.usage.sessionId.has_value());
}

// ---------------------------------------------------------------------------
// add() / accumulation semantics
// ---------------------------------------------------------------------------

TEST(LLMResponseAccumulator, AccumulatesTextAcrossChunksInOrder) {
  LLMResponseAccumulator acc;
  acc.add(makeChunkWithText("Hello, "));
  acc.add(makeChunkWithText("world"));
  acc.add(makeChunkWithText("!"));

  const LLMResponse response = acc.build(makeParams());
  EXPECT_EQ(response.choices[0].text, "Hello, world!");
  EXPECT_EQ(acc.tokenCount(), 3);
  EXPECT_EQ(response.usage.completion_tokens, 3);
  EXPECT_EQ(response.usage.total_tokens, PROMPT_TOKENS + 3);
}

TEST(LLMResponseAccumulator, AccumulatesReasoningWhenAnyChunkHasIt) {
  LLMResponseAccumulator acc;
  acc.add(makeChunkWithReasoning("Let me think. "));
  acc.add(makeChunkWithText("Answer"));
  acc.add(makeChunkWithReasoning("Done thinking."));

  const LLMResponse response = acc.build(makeParams());
  ASSERT_TRUE(response.choices[0].reasoning.has_value());
  EXPECT_EQ(response.choices[0].reasoning.value(),
            "Let me think. Done thinking.");
  EXPECT_EQ(response.choices[0].text, "Answer");
}

TEST(LLMResponseAccumulator, ReasoningStaysNulloptWhenNoChunkHasIt) {
  LLMResponseAccumulator acc;
  acc.add(makeChunkWithText("plain"));
  acc.add(makeChunkWithText(" answer"));

  const LLMResponse response = acc.build(makeParams());
  EXPECT_FALSE(response.choices[0].reasoning.has_value());
  EXPECT_EQ(response.choices[0].text, "plain answer");
}

TEST(LLMResponseAccumulator, IgnoresChunkWithEmptyChoices) {
  LLMResponseAccumulator acc;
  acc.add(makeChunkWithText("real "));
  acc.add(makeEmptyChoicesChunk());
  acc.add(makeChunkWithText("token"));

  const LLMResponse response = acc.build(makeParams());
  EXPECT_EQ(response.choices[0].text, "real token");
  EXPECT_EQ(acc.tokenCount(), 2);
}

// ---------------------------------------------------------------------------
// finish_reason handling
// ---------------------------------------------------------------------------

TEST(LLMResponseAccumulator, LastFinishReasonWins) {
  LLMResponseAccumulator acc;
  acc.add(makeFinishChunk("stop"));
  acc.add(makeFinishChunk("length"));
  acc.add(makeFinishChunk("error"));

  const LLMResponse response = acc.build(makeParams());
  ASSERT_TRUE(response.choices[0].finish_reason.has_value());
  EXPECT_EQ(response.choices[0].finish_reason.value(), "error");
}

TEST(LLMResponseAccumulator, FinishReasonFromChunkOverridesDefault) {
  LLMResponseAccumulator acc;
  acc.add(makeChunkWithText("output"));
  acc.add(makeFinishChunk("length"));

  const LLMResponse response = acc.build(makeParams());
  ASSERT_TRUE(response.choices[0].finish_reason.has_value());
  EXPECT_EQ(response.choices[0].finish_reason.value(), "length");
}

TEST(LLMResponseAccumulator, FinishOnlyChunkDoesNotCountAsToken) {
  LLMResponseAccumulator acc;
  acc.add(makeFinishChunk("stop"));
  EXPECT_EQ(acc.tokenCount(), 0);
}

// ---------------------------------------------------------------------------
// Token counting rules
// ---------------------------------------------------------------------------

TEST(LLMResponseAccumulator, EmptyTextWithoutReasoningSkipsTokenCount) {
  LLMResponseAccumulator acc;
  acc.add(makeChunkWithText(""));
  acc.add(makeChunkWithText(""));
  EXPECT_EQ(acc.tokenCount(), 0);
}

TEST(LLMResponseAccumulator, ReasoningOnlyChunkCountsAsToken) {
  LLMResponseAccumulator acc;
  acc.add(makeChunkWithReasoning("thinking"));
  EXPECT_EQ(acc.tokenCount(), 1);

  const LLMResponse response = acc.build(makeParams());
  EXPECT_EQ(response.usage.completion_tokens, 1);
  EXPECT_EQ(response.choices[0].text, "");
  ASSERT_TRUE(response.choices[0].reasoning.has_value());
  EXPECT_EQ(response.choices[0].reasoning.value(), "thinking");
}

TEST(LLMResponseAccumulator, MixedChunksTokenCountMatchesContentBearingChunks) {
  LLMResponseAccumulator acc;
  acc.add(makeChunkWithText("a"));         // counts
  acc.add(makeChunkWithText(""));          // skipped
  acc.add(makeChunkWithReasoning("r"));    // counts (reasoning-only)
  acc.add(makeEmptyChoicesChunk());        // skipped
  acc.add(makeChunkWithText("b"));         // counts
  acc.add(makeFinishChunk("stop"));        // skipped (no text/reasoning)
  EXPECT_EQ(acc.tokenCount(), 3);
}

// ---------------------------------------------------------------------------
// Idempotence / immutability of build()
// ---------------------------------------------------------------------------

TEST(LLMResponseAccumulator, BuildIsIdempotent) {
  LLMResponseAccumulator acc;
  acc.add(makeChunkWithText("hello"));
  acc.add(makeChunkWithReasoning("why"));
  acc.add(makeFinishChunk("length"));

  const LLMResponse first = acc.build(makeParams());
  const LLMResponse second = acc.build(makeParams());

  EXPECT_EQ(first.choices[0].text, second.choices[0].text);
  EXPECT_EQ(first.choices[0].reasoning, second.choices[0].reasoning);
  EXPECT_EQ(first.choices[0].finish_reason, second.choices[0].finish_reason);
  EXPECT_EQ(first.usage.completion_tokens, second.usage.completion_tokens);
  EXPECT_EQ(acc.tokenCount(), 1);
}

TEST(LLMResponseAccumulator, BuildPlacesSingleChoiceAtIndexZero) {
  LLMResponseAccumulator acc;
  acc.add(makeChunkWithText("answer"));
  const LLMResponse response = acc.build(makeParams());

  ASSERT_EQ(response.choices.size(), 1u);
  EXPECT_EQ(response.choices[0].index, 0);
}

// ---------------------------------------------------------------------------
// Usage totals and ttft sanity
// ---------------------------------------------------------------------------

TEST(LLMResponseAccumulator, UsageTotalIsPromptPlusCompletion) {
  LLMResponseAccumulator acc;
  acc.add(makeChunkWithText("a"));
  acc.add(makeChunkWithText("b"));

  const LLMResponse response = acc.build(makeParams());
  EXPECT_EQ(response.usage.prompt_tokens, PROMPT_TOKENS);
  EXPECT_EQ(response.usage.completion_tokens, 2);
  EXPECT_EQ(response.usage.total_tokens, PROMPT_TOKENS + 2);
}

TEST(LLMResponseAccumulator, TtftSetWhenAtLeastOneTokenAccumulated) {
  LLMResponseAccumulator acc;
  acc.add(makeChunkWithText("x"));
  const LLMResponse response = acc.build(makeParams());
  EXPECT_TRUE(response.usage.ttft_ms.has_value());
  EXPECT_GE(response.usage.ttft_ms.value(), 0.0);
}

TEST(LLMResponseAccumulator, TtftAbsentWhenNoTokens) {
  LLMResponseAccumulator acc;
  const LLMResponse response = acc.build(makeParams());
  EXPECT_FALSE(response.usage.ttft_ms.has_value());
  EXPECT_FALSE(response.usage.tps.has_value());
}

}  // namespace
