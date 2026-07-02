// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/native_prefill_handoff.hpp"

#include <gtest/gtest.h>

#include <string>

#include "domain/llm/llm_request.hpp"

namespace tt::dynamo {
namespace {

TEST(NativePrefillHandoffTest, MetadataOnlyHandoffIsValid) {
  auto handoff = buildMetadataOnlyNativePrefillHandoff(
      1234, 42, "prefill/generate/abc");

  Json::Value json = nativePrefillHandoffToJson(handoff);
  auto parsed = parseNativePrefillHandoff(json);

  EXPECT_EQ(parsed.selectedPrefillId, "prefill/generate/abc");
  EXPECT_EQ(parsed.migrationId, 1234u);
  EXPECT_EQ(parsed.kvPositionId, 41u);
  auto validation = validateNativePrefillHandoffForDecode(parsed);
  EXPECT_TRUE(validation.ok) << validation.error;
}

TEST(NativePrefillHandoffTest, FindsHandoffInDynamoPrefillResultEnvelope) {
  auto handoff = buildMetadataOnlyNativePrefillHandoff(
      1234, 42, "prefill/generate/abc");

  Json::Value raw(Json::objectValue);
  raw["prefill_result"]["disaggregated_params"] =
      nativePrefillHandoffToDisaggregatedParams(handoff);

  const Json::Value* found = findNativePrefillHandoffJson(raw);
  ASSERT_NE(found, nullptr);
  auto parsed = parseNativePrefillHandoff(*found);
  EXPECT_EQ(parsed.selectedPrefillId, "prefill/generate/abc");
  EXPECT_EQ(parsed.migrationId, 1234u);
}

TEST(NativePrefillHandoffTest, CompleteHandoffAppliesToRequest) {
  NativePrefillHandoff handoff;
  handoff.selectedPrefillId = "prefill/generate/abc";
  handoff.migrationId = 1234;
  handoff.kvPositionId = 41;
  handoff.decodeSlotId = 7;
  handoff.cachedTokens = 32;

  auto validation = validateNativePrefillHandoffForDecode(handoff);
  ASSERT_TRUE(validation.ok) << validation.error;

  tt::domain::llm::LLMRequest request(99);
  applyNativePrefillHandoffToRequest(handoff, request);

  EXPECT_TRUE(request.dynamoNativePrefillHandoff);
  EXPECT_TRUE(request.disaggregated);
  EXPECT_EQ(request.migrationId, 1234u);
  EXPECT_EQ(request.kv_position_id, 41u);
  EXPECT_EQ(request.dynamoNativePrefillDecodeSlotId, 7u);
  EXPECT_EQ(request.dynamoNativePrefillCachedTokens, 32);
}

TEST(NativePrefillHandoffTest, MissingSelectedPrefillFailsValidation) {
  NativePrefillHandoff handoff;
  handoff.migrationId = 1234;
  handoff.kvPositionId = 41;

  auto validation = validateNativePrefillHandoffForDecode(handoff);
  EXPECT_FALSE(validation.ok);
  EXPECT_NE(validation.error.find("selected_prefill_id"), std::string::npos);
}

}  // namespace
}  // namespace tt::dynamo
