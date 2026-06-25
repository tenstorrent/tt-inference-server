// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_runner/blaze_utils.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "domain/llm/sampling_params.hpp"
#include "domain/llm/sequence.hpp"

namespace {

TEST(BlazeUtilsTest,
     PrefillContinuationFallsBackToDecodePositionWhenKvPositionMissing) {
  constexpr uint32_t taskId = 123;
  constexpr uint32_t slotId = 5;
  constexpr uint32_t decodePositionId = 9984;

  std::vector<int64_t> replayTokens(24);
  for (size_t i = 0; i < replayTokens.size(); ++i) {
    replayTokens[i] = static_cast<int64_t>(decodePositionId + i);
  }

  tt::domain::llm::SamplingParams sampling;
  tt::domain::llm::Sequence seq(
      taskId, /*blockSize=*/32, std::move(replayTokens),
      /*numPromptTokens=*/24, /*slotId=*/slotId, /*prefillSlotId=*/slotId,
      /*continuation=*/true, /*disaggregated=*/false,
      std::make_unique<tt::domain::llm::SamplingParams>(sampling),
      /*kvPositionId=*/std::nullopt, /*decodePositionId=*/decodePositionId,
      /*decodeSkipTokens=*/decodePositionId, /*migrationId=*/42);

  const auto req = tt::runners::blaze::utils::makeSubmitRequest(
      slotId, seq, /*destSlotId=*/slotId);

  ASSERT_TRUE(req.position_id.has_value());
  EXPECT_EQ(*req.position_id, decodePositionId);
  EXPECT_EQ(req.tokens.size(), 24u);
}

}  // namespace
