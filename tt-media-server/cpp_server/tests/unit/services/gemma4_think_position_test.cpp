// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// The resume position a cached turn is continued from is
//     matchedTokens + accumulatedThinkTokens          (llm_pipeline.cpp)
// and it must equal the number of KV rows the matched prefix occupies.
//
// Thought delimiters are excluded from the block hash, which is correct: they
// are structure, and templates that strip thinking from the re-rendered history
// (gemma-4, deepseek-r1) do not carry them in the next prompt. But they are
// ordinary generated tokens and each occupies a KV row, so they must be counted.
// Before the fix they were excluded from both, leaving the resume position short
// by two rows per thought block and compounding over a conversation.
//
// Host-only: token ids, hashing and block arithmetic, no device.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "domain/prefix_cache/block_matcher.hpp"
#include "utils/conversation_hasher.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace {

constexpr uint32_t kBlockSize = 32;
constexpr uint32_t kContentBase = 1000;  // ordinary tokens, far from the markers

class Gemma4ThinkPositionTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // modelType() and the block sizes cache on first use.
    setenv("MODEL", "google/gemma-4-31B-it", 1);
    setenv("LLM_MODE", "regular", 1);
    setenv("KV_CACHE_FIRST_BLOCK_SIZE", "32", 1);
    setenv("KV_CACHE_BLOCK_SIZE", "32", 1);
  }
};

TEST_F(Gemma4ThinkPositionTest, ThoughtChannelDelimitersAreTheThinkTokenPair) {
  const auto [thinkStart, thinkEnd] = tt::utils::tokenizers::thinkTokenIds();
  EXPECT_EQ(thinkStart, 100u) << "<|channel>";
  EXPECT_EQ(thinkEnd, 101u) << "<channel|>";
}

TEST_F(Gemma4ThinkPositionTest, ResumePositionMatchesResidentKvRows) {
  const auto [thinkStart, thinkEnd] = tt::utils::tokenizers::thinkTokenIds();

  // 128 visible tokens = exactly 4 blocks at 32/32, so no partial-block
  // remainder muddies the arithmetic.
  constexpr uint32_t kVisible = 4 * kBlockSize;
  constexpr uint32_t kThinkContent = 10;
  constexpr uint32_t kMarkers = 2;

  std::vector<uint32_t> tokens;
  uint32_t next = kContentBase;
  for (uint32_t i = 0; i < 40; ++i) tokens.push_back(next++);
  tokens.push_back(thinkStart);
  for (uint32_t i = 0; i < kThinkContent; ++i) tokens.push_back(next++);
  tokens.push_back(thinkEnd);
  for (uint32_t i = 0; i < kVisible - 40; ++i) tokens.push_back(next++);

  const auto blocks = tt::utils::getPrefixCacheHashesByBlocksWithThinking(
      tokens, thinkStart, thinkEnd);
  ASSERT_FALSE(blocks.empty());

  const uint32_t matchedTokens =
      tt::domain::prefix_cache::BlockMatcher::blocksToTokens(blocks.size());
  const uint32_t thinkRows = blocks.back().accumulatedThinkTokens;

  EXPECT_EQ(matchedTokens, kVisible) << "markers and thinking stay out of the hash";
  EXPECT_EQ(thinkRows, kThinkContent + kMarkers) << "delimiters occupy rows too";
  EXPECT_EQ(matchedTokens + thinkRows, tokens.size())
      << "resume position must equal the rows the prefix occupies";
}

TEST_F(Gemma4ThinkPositionTest, DriftDoesNotAccumulateAcrossThoughtBlocks) {
  const auto [thinkStart, thinkEnd] = tt::utils::tokenizers::thinkTokenIds();
  constexpr uint32_t kVisiblePerTurn = 2 * kBlockSize;
  constexpr uint32_t kThinkPerTurn = 8;

  for (uint32_t turns : {1u, 2u, 3u, 8u}) {
    std::vector<uint32_t> tokens;
    uint32_t next = kContentBase;
    for (uint32_t t = 0; t < turns; ++t) {
      for (uint32_t i = 0; i < kVisiblePerTurn; ++i) tokens.push_back(next++);
      tokens.push_back(thinkStart);
      for (uint32_t i = 0; i < kThinkPerTurn; ++i) tokens.push_back(next++);
      tokens.push_back(thinkEnd);
    }

    const auto blocks = tt::utils::getPrefixCacheHashesByBlocksWithThinking(
        tokens, thinkStart, thinkEnd);
    ASSERT_FALSE(blocks.empty());
    const uint32_t resume =
        tt::domain::prefix_cache::BlockMatcher::blocksToTokens(blocks.size()) +
        blocks.back().accumulatedThinkTokens;

    EXPECT_EQ(resume, tokens.size())
        << "after " << turns << " thought block(s) the resume position is short by "
        << (tokens.size() - resume) << " row(s)";
  }
}

// Thinking disabled: no markers, no correction, position is the token count.
TEST_F(Gemma4ThinkPositionTest, NoThinkingLeavesTheCountAtZero) {
  std::vector<uint32_t> tokens;
  for (uint32_t i = 0; i < 4 * kBlockSize; ++i) tokens.push_back(kContentBase + i);

  const auto blocks = tt::utils::getPrefixCacheHashesByBlocksWithThinking(
      tokens, tt::utils::tokenizers::kNoTokenId,
      tt::utils::tokenizers::kNoTokenId);
  ASSERT_FALSE(blocks.empty());
  EXPECT_EQ(blocks.back().accumulatedThinkTokens, 0u);
  EXPECT_EQ(tt::domain::prefix_cache::BlockMatcher::blocksToTokens(blocks.size()),
            tokens.size());
}

}  // namespace
