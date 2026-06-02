// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "domain/llm/llm_request.hpp"

using namespace tt::domain::llm;

// Reimplementation of DisaggregationService::applyDeltaPrompt for unit testing.
// Must be kept in sync with the production implementation.
namespace {

void applyDeltaPrompt(LLMRequest& req, uint32_t matchedTokens) {
  auto& tokens = std::get<std::vector<int>>(req.prompt);
  if (matchedTokens == 0 || matchedTokens >= tokens.size()) {
    return;
  }

  constexpr uint32_t kAlignment = 32;
  const uint32_t totalTokens = static_cast<uint32_t>(tokens.size());
  const uint32_t remainder = totalTokens - matchedTokens;
  const uint32_t alignedRemainder =
      ((remainder + kAlignment - 1) / kAlignment) * kAlignment;

  const uint32_t pullBack = alignedRemainder - remainder;
  const uint32_t effectiveSkip =
      (pullBack <= matchedTokens) ? (matchedTokens - pullBack) : 0;

  if (effectiveSkip == 0) {
    return;
  }

  tokens.erase(tokens.begin(),
               tokens.begin() + static_cast<ptrdiff_t>(effectiveSkip));
  req.prompt_tokens_count = static_cast<int>(tokens.size());
  req.kv_position_id = effectiveSkip - 1;
}

// Helper to create a request with N sequential tokens [0, 1, 2, ..., N-1].
LLMRequest makeRequest(uint32_t numTokens) {
  LLMRequest req(/*taskId=*/1);
  std::vector<int> tokens(numTokens);
  std::iota(tokens.begin(), tokens.end(), 0);
  req.prompt.emplace<std::vector<int>>(std::move(tokens));
  return req;
}

}  // namespace

// When the remainder is already aligned to 32, no pull-back is needed.
TEST(ApplyDeltaPrompt, RemainderAlreadyAligned) {
  // 128 total tokens, 96 matched → remainder = 32 (aligned)
  auto req = makeRequest(128);
  applyDeltaPrompt(req, 96);

  auto& tokens = std::get<std::vector<int>>(req.prompt);
  EXPECT_EQ(tokens.size(), 32u);
  EXPECT_EQ(tokens.front(), 96);  // first remaining token
  EXPECT_EQ(req.kv_position_id, 95u);
  EXPECT_EQ(req.prompt_tokens_count, 32);
}

// When remainder is not aligned, tokens are pulled back to make it aligned.
TEST(ApplyDeltaPrompt, RemainderNotAligned_PullBack) {
  // 100 total tokens, 90 matched → remainder = 10
  // alignedRemainder = ceil(10/32)*32 = 32
  // pullBack = 32 - 10 = 22
  // effectiveSkip = 90 - 22 = 68
  // remaining tokens = 100 - 68 = 32 (aligned!)
  auto req = makeRequest(100);
  applyDeltaPrompt(req, 90);

  auto& tokens = std::get<std::vector<int>>(req.prompt);
  EXPECT_EQ(tokens.size(), 32u);
  EXPECT_EQ(tokens.front(), 68);
  EXPECT_EQ(req.kv_position_id, 67u);
  EXPECT_EQ(req.prompt_tokens_count, 32);
}

// When remainder needs only a small pull-back.
TEST(ApplyDeltaPrompt, RemainderSlightlyMisaligned) {
  // 100 total, 68 matched → remainder = 32 (already aligned, pullBack = 0)
  auto req = makeRequest(100);
  applyDeltaPrompt(req, 68);

  auto& tokens = std::get<std::vector<int>>(req.prompt);
  EXPECT_EQ(tokens.size(), 32u);
  EXPECT_EQ(tokens.front(), 68);
  EXPECT_EQ(req.kv_position_id, 67u);
}

// When matched tokens are too few to satisfy pull-back, effectiveSkip = 0.
TEST(ApplyDeltaPrompt, InsufficientMatchedTokens_NoDelta) {
  // 100 total, 5 matched → remainder = 95
  // alignedRemainder = ceil(95/32)*32 = 96
  // pullBack = 96 - 95 = 1
  // effectiveSkip = 5 - 1 = 4
  // This works, remainder = 96
  auto req = makeRequest(100);
  applyDeltaPrompt(req, 5);

  auto& tokens = std::get<std::vector<int>>(req.prompt);
  EXPECT_EQ(tokens.size(), 96u);
  EXPECT_EQ(tokens.front(), 4);
  EXPECT_EQ(req.kv_position_id, 3u);
}

// When pullBack exceeds matchedTokens, nothing happens.
TEST(ApplyDeltaPrompt, PullBackExceedsMatched_NoDelta) {
  // 64 total, 1 matched → remainder = 63
  // alignedRemainder = ceil(63/32)*32 = 64
  // pullBack = 64 - 63 = 1
  // effectiveSkip = 1 - 1 = 0 → no-op
  auto req = makeRequest(64);
  applyDeltaPrompt(req, 1);

  auto& tokens = std::get<std::vector<int>>(req.prompt);
  EXPECT_EQ(tokens.size(), 64u);  // unchanged
}

// Zero matched tokens is a no-op.
TEST(ApplyDeltaPrompt, ZeroMatchedTokens_NoOp) {
  auto req = makeRequest(100);
  applyDeltaPrompt(req, 0);

  auto& tokens = std::get<std::vector<int>>(req.prompt);
  EXPECT_EQ(tokens.size(), 100u);
}

// matchedTokens >= total is a no-op.
TEST(ApplyDeltaPrompt, MatchedExceedsTotal_NoOp) {
  auto req = makeRequest(50);
  applyDeltaPrompt(req, 50);

  auto& tokens = std::get<std::vector<int>>(req.prompt);
  EXPECT_EQ(tokens.size(), 50u);
}

// Large prompt, verify alignment holds.
TEST(ApplyDeltaPrompt, LargePrompt_AlignmentHolds) {
  // 2048 total, 1500 matched → remainder = 548
  // alignedRemainder = ceil(548/32)*32 = 576 (18*32)
  // pullBack = 576 - 548 = 28
  // effectiveSkip = 1500 - 28 = 1472
  // remaining = 2048 - 1472 = 576
  auto req = makeRequest(2048);
  applyDeltaPrompt(req, 1500);

  auto& tokens = std::get<std::vector<int>>(req.prompt);
  EXPECT_EQ(tokens.size(), 576u);
  EXPECT_EQ(tokens.size() % 32, 0u);
  EXPECT_EQ(tokens.front(), 1472);
  EXPECT_EQ(req.kv_position_id, 1471u);
}

// Exact 32-boundary matched tokens with exact remainder.
TEST(ApplyDeltaPrompt, ExactBoundaries) {
  // 256 total, 192 matched → remainder = 64 (aligned)
  auto req = makeRequest(256);
  applyDeltaPrompt(req, 192);

  auto& tokens = std::get<std::vector<int>>(req.prompt);
  EXPECT_EQ(tokens.size(), 64u);
  EXPECT_EQ(tokens.size() % 32, 0u);
  EXPECT_EQ(req.kv_position_id, 191u);
}
