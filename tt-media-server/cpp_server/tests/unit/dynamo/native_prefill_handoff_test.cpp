// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/native_prefill_handoff.hpp"

#include <gtest/gtest.h>

#include <string>

#include "domain/llm/llm_request.hpp"

namespace tt::dynamo {
namespace {

TEST(NativePrefillHandoffTest, MetadataOnlyHandoffIsIncomplete) {
  auto handoff = buildMetadataOnlyNativePrefillHandoff(
      "req-1", 1234, 42, "prefill/generate/abc");

  Json::Value json = nativePrefillHandoffToJson(handoff);
  auto parsed = parseNativePrefillHandoff(json);

  EXPECT_EQ(parsed.selected_prefill_id, "prefill/generate/abc");
  EXPECT_EQ(parsed.migration_id, 1234u);
  EXPECT_EQ(parsed.kv_position_id, 41u);
  EXPECT_EQ(parsed.mooncake_status, "not_started");

  auto validation = validateNativePrefillHandoffForDecode(parsed);
  EXPECT_FALSE(validation.ok);
}

TEST(NativePrefillHandoffTest, FindsHandoffInDynamoPrefillResultEnvelope) {
  auto handoff = buildMetadataOnlyNativePrefillHandoff(
      "req-1", 1234, 42, "prefill/generate/abc");

  Json::Value raw(Json::objectValue);
  raw["prefill_result"]["disaggregated_params"] =
      nativePrefillHandoffToDisaggregatedParams(handoff);

  const Json::Value* found = findNativePrefillHandoffJson(raw);
  ASSERT_NE(found, nullptr);
  auto parsed = parseNativePrefillHandoff(*found);
  EXPECT_EQ(parsed.selected_prefill_id, "prefill/generate/abc");
  EXPECT_EQ(parsed.migration_id, 1234u);
}

TEST(NativePrefillHandoffTest, CompleteHandoffAppliesToRequest) {
  NativePrefillHandoff handoff;
  handoff.selected_prefill_id = "prefill/generate/abc";
  handoff.migration_id = 1234;
  handoff.kv_position_id = 41;
  handoff.decode_slot_id = 7;
  handoff.cached_tokens = 32;
  handoff.mooncake_uuid = 1234;
  handoff.mooncake_status = "complete";

  auto validation = validateNativePrefillHandoffForDecode(handoff);
  ASSERT_TRUE(validation.ok) << validation.error;

  tt::domain::llm::LLMRequest request(99);
  applyNativePrefillHandoffToRequest(handoff, request);

  EXPECT_TRUE(request.dynamo_native_prefill_handoff);
  EXPECT_TRUE(request.disaggregated);
  EXPECT_EQ(request.migrationId, 1234u);
  EXPECT_EQ(request.kv_position_id, 41u);
  EXPECT_EQ(request.dynamo_native_prefill_decode_slot_id, 7u);
  EXPECT_EQ(request.dynamo_native_prefill_cached_tokens, 32);
}

TEST(NativePrefillHandoffTest, MissingSelectedPrefillFailsValidation) {
  NativePrefillHandoff handoff;
  handoff.migration_id = 1234;
  handoff.kv_position_id = 41;
  handoff.mooncake_status = "complete";

  auto validation = validateNativePrefillHandoffForDecode(handoff);
  EXPECT_FALSE(validation.ok);
  EXPECT_NE(validation.error.find("selected_prefill_id"), std::string::npos);
}

}  // namespace
}  // namespace tt::dynamo
