// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/response_writer/response_writer.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

namespace {

using tt::api::ResponseWriter;
using tt::api::ResponseWriterParams;
using tt::domain::llm::CompletionUsage;
using tt::domain::llm::LLMStreamChunk;

// Minimal concrete subclass that exposes the protected helpers and
// no-ops the abstract methods. The base class is the unit under test;
// this fake exists purely to instantiate it and reach the protected API.
class TestableWriter : public ResponseWriter {
 public:
  explicit TestableWriter(ResponseWriterParams params)
      : ResponseWriter(std::move(params)) {}

  void handleTokenChunk(const LLMStreamChunk&) override {}
  void finalize() override { done.store(true); }

  using ResponseWriter::buildUsage;
  using ResponseWriter::noteToken;
  using ResponseWriter::releaseInFlight;
};

constexpr uint32_t TASK_ID = 99;
constexpr int PROMPT_TOKENS = 5;

ResponseWriterParams baseParams() {
  ResponseWriterParams params;
  params.completionId = "chatcmpl-rw";
  params.model = "rw-model";
  params.created = 1700000000;
  params.promptTokenCount = PROMPT_TOKENS;
  params.taskId = TASK_ID;
  return params;
}

// ---------------------------------------------------------------------------
// releaseInFlight: lambda-based release plumbing (the refactored path)
// ---------------------------------------------------------------------------

TEST(ResponseWriterReleaseInFlight, InvokesProvidedLambdaExactlyOnce) {
  auto params = baseParams();
  std::atomic<int> calls{0};
  params.releaseInFlightFn = [&calls]() { calls.fetch_add(1); };

  TestableWriter writer(std::move(params));
  EXPECT_EQ(calls.load(), 0);

  writer.releaseInFlight();
  EXPECT_EQ(calls.load(), 1);
}

TEST(ResponseWriterReleaseInFlight, IsNoOpWhenLambdaIsNull) {
  TestableWriter writer(baseParams());
  EXPECT_NO_THROW(writer.releaseInFlight());
}

TEST(ResponseWriterReleaseInFlight, RepeatedCallsForwardToLambdaEachTime) {
  // The base class doesn't deduplicate releases; idempotency is the
  // controller's responsibility (or finalize()'s done flag). Test the
  // wiring contract: every call hits the lambda.
  auto params = baseParams();
  std::atomic<int> calls{0};
  params.releaseInFlightFn = [&calls]() { calls.fetch_add(1); };

  TestableWriter writer(std::move(params));
  writer.releaseInFlight();
  writer.releaseInFlight();
  writer.releaseInFlight();
  EXPECT_EQ(calls.load(), 3);
}

// ---------------------------------------------------------------------------
// noteToken: counter and timing capture
// ---------------------------------------------------------------------------

TEST(ResponseWriterNoteToken, IncrementsCounterAndReturnsNewValue) {
  TestableWriter writer(baseParams());
  EXPECT_EQ(writer.noteToken(), 1);
  EXPECT_EQ(writer.noteToken(), 2);
  EXPECT_EQ(writer.noteToken(), 3);
}

TEST(ResponseWriterNoteToken, FirstTokenSetsTtftAfterAtLeastOneCall) {
  TestableWriter writer(baseParams());
  CompletionUsage before = writer.buildUsage();
  EXPECT_FALSE(before.ttft_ms.has_value());

  writer.noteToken();
  CompletionUsage after = writer.buildUsage();
  EXPECT_TRUE(after.ttft_ms.has_value());
  EXPECT_GE(after.ttft_ms.value(), 0.0);
}

TEST(ResponseWriterNoteToken, TpsAvailableAfterTwoTokensWithDelay) {
  TestableWriter writer(baseParams());
  writer.noteToken();
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  writer.noteToken();
  std::this_thread::sleep_for(std::chrono::milliseconds(2));

  CompletionUsage usage = writer.buildUsage();
  EXPECT_EQ(usage.completion_tokens, 2);
  EXPECT_TRUE(usage.tps.has_value());
  EXPECT_GT(usage.tps.value(), 0.0);
}

// ---------------------------------------------------------------------------
// buildUsage: prompt/total wiring and sessionId propagation
// ---------------------------------------------------------------------------

TEST(ResponseWriterBuildUsage, PropagatesPromptTokensAndComputesTotal) {
  TestableWriter writer(baseParams());
  writer.noteToken();
  writer.noteToken();

  CompletionUsage usage = writer.buildUsage();
  EXPECT_EQ(usage.prompt_tokens, PROMPT_TOKENS);
  EXPECT_EQ(usage.completion_tokens, 2);
  EXPECT_EQ(usage.total_tokens, PROMPT_TOKENS + 2);
}

TEST(ResponseWriterBuildUsage, IncludesSessionIdWhenSet) {
  auto params = baseParams();
  params.sessionId = "sess-xyz";

  TestableWriter writer(std::move(params));
  CompletionUsage usage = writer.buildUsage();
  ASSERT_TRUE(usage.sessionId.has_value());
  EXPECT_EQ(usage.sessionId.value(), "sess-xyz");
}

TEST(ResponseWriterBuildUsage, OmitsSessionIdWhenUnset) {
  TestableWriter writer(baseParams());
  CompletionUsage usage = writer.buildUsage();
  EXPECT_FALSE(usage.sessionId.has_value());
}

// ---------------------------------------------------------------------------
// done flag
// ---------------------------------------------------------------------------

TEST(ResponseWriterDoneFlag, FalseInitiallyAndTrueAfterFinalize) {
  TestableWriter writer(baseParams());
  EXPECT_FALSE(writer.isDone());
  writer.finalize();
  EXPECT_TRUE(writer.isDone());
}

}  // namespace
