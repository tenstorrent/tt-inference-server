// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "domain/llm/llm_request.hpp"
#include "services/session_resolution.hpp"

using namespace tt::domain::llm;

// Exercise the production helper used by both DisaggregationService and
// LLMPipeline session resolution.
//
// `matchedTokens` is always a multiple of 32 (prefix-cache blocks are 32 tokens
// wide), so we trim exactly `matchedTokens` and the resumed prefill starts on a
// tile boundary. No 32-rounding / pull-back of matched tokens is performed; the
// trailing partial tile of the remaining suffix is handled by prefill.
namespace {

void applyDeltaPrompt(LLMRequest& req, uint32_t matchedTokens) {
  tt::services::session_resolution::applyDeltaPrompt(
      req, matchedTokens,
      {.skipUnlessRegularMode = false,
       .setKvPositionId = true,
       .logPrefix = {}});
}

// Helper to create a request with N sequential tokens [0, 1, 2, ..., N-1].
LLMRequest makeRequest(uint32_t numTokens) {
  LLMRequest req(/*taskId=*/1);
  std::vector<uint32_t> tokens(numTokens);
  std::iota(tokens.begin(), tokens.end(), 0);
  req.prompt.emplace<std::vector<uint32_t>>(std::move(tokens));
  return req;
}

}  // namespace

// Matched tokens are trimmed exactly; an already-aligned remainder is
// untouched.
TEST(ApplyDeltaPrompt, RemainderAlreadyAligned) {
  // 128 total tokens, 96 matched → remainder = 32
  auto req = makeRequest(128);
  applyDeltaPrompt(req, 96);

  auto& tokens = std::get<std::vector<uint32_t>>(req.prompt);
  EXPECT_EQ(tokens.size(), 32u);
  EXPECT_EQ(tokens.front(), 96);       // first remaining token
  EXPECT_EQ(req.kv_position_id, 96u);  // first free KV index
  EXPECT_EQ(req.prompt_tokens_count, 32);
}

// A non-32-aligned remainder is sent as-is (no pull-back): the resumed prefill
// still starts on a tile boundary because `matchedTokens` is 32-aligned, and
// the trailing partial tile is fine for prefill.
TEST(ApplyDeltaPrompt, RemainderNotAligned_SentAsIs) {
  // 657 total tokens, 640 matched → remainder = 17 (partial tile)
  auto req = makeRequest(657);
  applyDeltaPrompt(req, 640);

  auto& tokens = std::get<std::vector<uint32_t>>(req.prompt);
  EXPECT_EQ(tokens.size(), 17u);
  EXPECT_EQ(tokens.front(), 640);
  EXPECT_EQ(req.kv_position_id, 640u);
  EXPECT_EQ(req.prompt_tokens_count, 17);
}

// The whole remaining suffix (including the prior partial tile) is sent.
TEST(ApplyDeltaPrompt, MultiTurnSuffix) {
  // Scenario: 640 cached (32-aligned) of a (17 + 640 + 1280) = 1937 prompt.
  auto req = makeRequest(1937);
  applyDeltaPrompt(req, 640);

  auto& tokens = std::get<std::vector<uint32_t>>(req.prompt);
  EXPECT_EQ(tokens.size(), 1297u);  // 17 + 1280
  EXPECT_EQ(tokens.front(), 640);
  EXPECT_EQ(req.kv_position_id, 640u);
  EXPECT_EQ(req.prompt_tokens_count, 1297);
}

// matchedTokens >= total is a no-op.
TEST(ApplyDeltaPrompt, MatchedExceedsTotal_NoOp) {
  auto req = makeRequest(50);
  applyDeltaPrompt(req, 50);

  auto& tokens = std::get<std::vector<uint32_t>>(req.prompt);
  EXPECT_EQ(tokens.size(), 50u);
  EXPECT_FALSE(req.kv_position_id.has_value());
}

// Large prompt: trims exactly the matched prefix, suffix kept verbatim.
TEST(ApplyDeltaPrompt, LargePrompt) {
  // 2048 total, 1504 matched (47 * 32) → remainder = 544
  auto req = makeRequest(2048);
  applyDeltaPrompt(req, 1504);

  auto& tokens = std::get<std::vector<uint32_t>>(req.prompt);
  EXPECT_EQ(tokens.size(), 544u);
  EXPECT_EQ(tokens.front(), 1504);
  EXPECT_EQ(req.kv_position_id, 1504u);
}

// Exact 32-boundary matched tokens with exact remainder.
TEST(ApplyDeltaPrompt, ExactBoundaries) {
  // 256 total, 192 matched → remainder = 64
  auto req = makeRequest(256);
  applyDeltaPrompt(req, 192);

  auto& tokens = std::get<std::vector<uint32_t>>(req.prompt);
  EXPECT_EQ(tokens.size(), 64u);
  EXPECT_EQ(req.kv_position_id, 192u);
}
