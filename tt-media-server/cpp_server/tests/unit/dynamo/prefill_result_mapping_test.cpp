// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "dynamo/prefill_result_mapping.hpp"

#include <gtest/gtest.h>
#include <json/json.h>

#include <memory>
#include <string>

namespace {

constexpr const char* kTraceparent =
    "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

Json::Value parseJsonString(const std::string& body) {
  Json::Value root;
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  std::string errors;
  EXPECT_TRUE(
      reader->parse(body.data(), body.data() + body.size(), &root, &errors))
      << errors;
  return root;
}

}  // namespace

TEST(PrefillResultMappingTest, TraceparentAbsentWhenNoPrefillResult) {
  const Json::Value raw = parseJsonString(R"({"model": "m", "token_ids": []})");
  EXPECT_EQ(tt::dynamo::prefillResultTraceparent(raw), "");
  EXPECT_FALSE(tt::dynamo::prefillResultFromJson(raw).has_value());
}

TEST(PrefillResultMappingTest, TraceparentFromDisaggregatedParams) {
  const Json::Value raw = parseJsonString(
      R"({"disaggregated_params": {"tt_prefill_result": {"task_id": 7,)"
      R"("traceparent": ")" +
      std::string(kTraceparent) + R"("}}})");
  EXPECT_EQ(tt::dynamo::prefillResultTraceparent(raw), kTraceparent);
}

TEST(PrefillResultMappingTest, TraceparentFromPrefillResultNesting) {
  const Json::Value raw = parseJsonString(
      R"({"prefill_result": {"disaggregated_params": {"tt_prefill_result":)"
      R"({"task_id": 7, "traceparent": ")" +
      std::string(kTraceparent) + R"("}}}})");
  EXPECT_EQ(tt::dynamo::prefillResultTraceparent(raw), kTraceparent);
}

TEST(PrefillResultMappingTest, TraceparentFromExtraArgsNesting) {
  const Json::Value raw = parseJsonString(
      R"({"extra_args": {"prefill_result": {"disaggregated_params":)"
      R"({"tt_prefill_result": {"task_id": 7, "traceparent": ")" +
      std::string(kTraceparent) + R"("}}}}})");
  EXPECT_EQ(tt::dynamo::prefillResultTraceparent(raw), kTraceparent);
}

TEST(PrefillResultMappingTest, TraceparentIgnoredWhenNotAString) {
  const Json::Value raw = parseJsonString(
      R"({"disaggregated_params": {"tt_prefill_result": {"task_id": 7,)"
      R"("traceparent": 42}}})");
  EXPECT_EQ(tt::dynamo::prefillResultTraceparent(raw), "");
}

TEST(PrefillResultMappingTest, ResultParsingUnaffectedByTraceparentField) {
  const Json::Value raw = parseJsonString(
      R"({"disaggregated_params": {"tt_prefill_result": {"task_id": 7,)"
      R"("generated_text": "hi", "token_ids": [1, 2], "traceparent": ")" +
      std::string(kTraceparent) + R"("}}})");

  const auto result = tt::dynamo::prefillResultFromJson(raw);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->taskId, 7u);
  EXPECT_EQ(result->generatedText, "hi");
  EXPECT_EQ(result->tokenIds.size(), 2u);
  EXPECT_EQ(tt::dynamo::prefillResultTraceparent(raw), kTraceparent);
}
