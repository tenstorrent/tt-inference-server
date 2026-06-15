// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "utils/tokenizers/thinking_phase.hpp"

#include <gtest/gtest.h>

#include <span>
#include <vector>

#include "domain/llm/llm_request.hpp"

namespace tt::utils::tokenizers {
namespace {

constexpr int64_t kThinkOpen = 100;
constexpr int64_t kThinkClose = 101;

TEST(ThinkingPhaseTest, TrailingOpenMarker) {
  const std::vector<int> prompt = {1, 2, 3, static_cast<int>(kThinkOpen)};
  EXPECT_TRUE(computeThinkingPhaseFromTokens(
      false, std::span<const int>(prompt), kThinkOpen, kThinkClose));
}

TEST(ThinkingPhaseTest, ClosedBlockEndsNonThinking) {
  const std::vector<int> prompt = {1, static_cast<int>(kThinkOpen), 7,
                                   static_cast<int>(kThinkClose), 9};
  EXPECT_FALSE(computeThinkingPhaseFromTokens(
      false, std::span<const int>(prompt), kThinkOpen, kThinkClose));
}

TEST(ThinkingPhaseTest, ResumeInsideBlockPreservedWhenNoMarker) {
  const std::vector<int> delta = {5, 6, 7};
  EXPECT_TRUE(computeThinkingPhaseFromTokens(true, std::span<const int>(delta),
                                             kThinkOpen, kThinkClose));
}

TEST(ThinkingPhaseTest, RefreshStartsInThinkingOnLLMRequest) {
  const auto [thinkOpen, thinkEnd] = thinkTokenIds();
  if (thinkOpen == kNoTokenId) {
    GTEST_SKIP() << "Active model has no think markers configured";
  }
  tt::domain::llm::LLMRequest req(1);
  req.prompt = std::vector<int>{5, 6, static_cast<int>(thinkOpen)};
  refreshStartsInThinking(req);
  EXPECT_TRUE(req.starts_in_thinking);
}

TEST(ThinkingPhaseTest, ClampedScanIgnoresTruncatedMarker) {
  const std::vector<int> prompt = {1, 2, 3, static_cast<int>(kThinkOpen)};
  EXPECT_FALSE(computeThinkingPhaseFromTokens(
      false, std::span<const int>(prompt).first(3), kThinkOpen, kThinkClose));
}

}  // namespace
}  // namespace tt::utils::tokenizers
