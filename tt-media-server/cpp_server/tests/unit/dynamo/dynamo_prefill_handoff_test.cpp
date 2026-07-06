// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/dynamo_prefill_handoff.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "sockets/socket_messages.hpp"

namespace tt::dynamo {
namespace {

tt::sockets::PrefillResultMessage makePrefillResult() {
  tt::sockets::PrefillResultMessage result(99);
  result.generatedText = "x";
  result.tokenIds = {10, 20, 30};
  result.remainingTokens = 4;
  result.slotId = 7;
  result.temperature = 0.5f;
  result.topP = 0.9f;
  result.topK = 32;
  result.fastMode = true;
  result.cachedTokens = 2;
  result.migrationId = 1234;
  return result;
}

TEST(DynamoPrefillHandoffTest, PrefillResultHandoffIsValid) {
  auto handoff = dynamoPrefillHandoffFromPrefillResult(
      makePrefillResult(), "prefill/generate/abc");

  Json::Value json = dynamoPrefillHandoffToJson(handoff);
  auto parsed = parseDynamoPrefillHandoff(json);

  EXPECT_EQ(parsed.selectedPrefillId, "prefill/generate/abc");
  ASSERT_TRUE(parsed.migrationId.has_value());
  EXPECT_EQ(*parsed.migrationId, 1234u);
  ASSERT_TRUE(parsed.kvPositionId.has_value());
  EXPECT_EQ(*parsed.kvPositionId, 2u);
  ASSERT_TRUE(parsed.decodeSlotId.has_value());
  EXPECT_EQ(*parsed.decodeSlotId, 7u);
  EXPECT_EQ(parsed.tokenIds, std::vector<int64_t>({10, 20, 30}));
  ASSERT_TRUE(parsed.remainingTokens.has_value());
  EXPECT_EQ(*parsed.remainingTokens, 4);
  EXPECT_EQ(parsed.cachedTokens, 2);
  EXPECT_TRUE(parsed.fastMode);
  auto validation = validateDynamoPrefillHandoffForDecode(parsed);
  EXPECT_TRUE(validation.ok) << validation.error;
}

TEST(DynamoPrefillHandoffTest, FindsHandoffInDynamoPrefillResultEnvelope) {
  auto handoff = dynamoPrefillHandoffFromPrefillResult(
      makePrefillResult(), "prefill/generate/abc");

  Json::Value raw(Json::objectValue);
  raw["prefill_result"]["disaggregated_params"] =
      dynamoPrefillHandoffToDisaggregatedParams(handoff);

  const Json::Value* found = findDynamoPrefillHandoffJson(raw);
  ASSERT_NE(found, nullptr);
  auto parsed = parseDynamoPrefillHandoff(*found);
  EXPECT_EQ(parsed.selectedPrefillId, "prefill/generate/abc");
  ASSERT_TRUE(parsed.migrationId.has_value());
  EXPECT_EQ(*parsed.migrationId, 1234u);
}

TEST(DynamoPrefillHandoffTest, HandoffConvertsBackToPrefillResult) {
  auto handoff = dynamoPrefillHandoffFromPrefillResult(
      makePrefillResult(), "prefill/generate/abc");

  auto result = dynamoPrefillHandoffToPrefillResult(100, handoff);

  EXPECT_EQ(result.taskId, 100u);
  EXPECT_EQ(result.generatedText, "x");
  EXPECT_EQ(result.tokenIds, std::vector<int64_t>({10, 20, 30}));
  ASSERT_TRUE(result.remainingTokens.has_value());
  EXPECT_EQ(*result.remainingTokens, 4);
  ASSERT_TRUE(result.slotId.has_value());
  EXPECT_EQ(*result.slotId, 7u);
  ASSERT_TRUE(result.temperature.has_value());
  EXPECT_EQ(*result.temperature, 0.5f);
  ASSERT_TRUE(result.topP.has_value());
  EXPECT_EQ(*result.topP, 0.9f);
  ASSERT_TRUE(result.topK.has_value());
  EXPECT_EQ(*result.topK, 32);
  EXPECT_TRUE(result.fastMode);
  EXPECT_EQ(result.cachedTokens, 2);
  EXPECT_EQ(result.migrationId, 1234u);
}

TEST(DynamoPrefillHandoffTest, MissingSelectedPrefillFailsValidation) {
  DynamoPrefillHandoff handoff;
  handoff.migrationId = 1234;
  handoff.kvPositionId = 41;

  auto validation = validateDynamoPrefillHandoffForDecode(handoff);
  EXPECT_FALSE(validation.ok);
  EXPECT_NE(validation.error.find("selected_prefill_id"), std::string::npos);
}

}  // namespace
}  // namespace tt::dynamo
