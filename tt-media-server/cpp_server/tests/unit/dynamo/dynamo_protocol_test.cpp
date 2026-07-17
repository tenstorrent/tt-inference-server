// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>
#include <json/json.h>

#include <memory>
#include <string>
#include <vector>

#include "dynamo/transport/protocol.hpp"

namespace {

Json::Value parseJson(const std::vector<uint8_t>& bytes) {
  Json::Value root;
  Json::CharReaderBuilder builder;
  builder["collectComments"] = false;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  std::string errors;
  const std::string body(bytes.begin(), bytes.end());
  EXPECT_TRUE(
      reader->parse(body.data(), body.data() + body.size(), &root, &errors))
      << errors;
  return root;
}

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

TEST(DynamoProtocolTest, ErrorChunkUsesAnnotatedErrorNotFinishReasonError) {
  tt::dynamo::TokenChunk chunk;
  chunk.error = "prefill error";
  chunk.error_code = 500;

  const Json::Value wrapper = parseJson(tt::dynamo::encode_stream_chunk(chunk));

  ASSERT_TRUE(wrapper.isObject());
  EXPECT_FALSE(wrapper["complete_final"].asBool());
  ASSERT_TRUE(wrapper["data"].isObject());

  const Json::Value& annotated = wrapper["data"];
  EXPECT_TRUE(annotated["data"].isNull());
  EXPECT_EQ(annotated["event"].asString(), "error");
  ASSERT_TRUE(annotated["comment"].isArray());
  ASSERT_EQ(annotated["comment"].size(), 1u);

  const Json::Value errorPayload =
      parseJsonString(annotated["comment"][0].asString());
  EXPECT_EQ(errorPayload["message"].asString(), "prefill error");
  EXPECT_EQ(errorPayload["code"].asInt(), 500);
  EXPECT_FALSE(annotated.isMember("finish_reason"));
}

TEST(DynamoProtocolTest, FinishReasonErrorUsesAnnotatedError) {
  tt::dynamo::TokenChunk chunk;
  chunk.finish_reason = "error";

  const Json::Value wrapper = parseJson(tt::dynamo::encode_stream_chunk(chunk));

  ASSERT_TRUE(wrapper.isObject());
  EXPECT_FALSE(wrapper["complete_final"].asBool());
  ASSERT_TRUE(wrapper["data"].isObject());

  const Json::Value& annotated = wrapper["data"];
  EXPECT_TRUE(annotated["data"].isNull())
      << "Dynamo should not receive finish_reason=\"error\" as a normal "
         "BackendOutput payload";
  EXPECT_EQ(annotated["event"].asString(), "error");
  ASSERT_TRUE(annotated["comment"].isArray());
  ASSERT_EQ(annotated["comment"].size(), 1u);

  const Json::Value errorPayload =
      parseJsonString(annotated["comment"][0].asString());
  EXPECT_EQ(errorPayload["message"].asString(), "error");
  EXPECT_EQ(errorPayload["code"].asInt(), 500);
}

TEST(DynamoProtocolTest, TimeoutFinishReasonUsesAnnotatedError) {
  tt::dynamo::TokenChunk chunk;
  chunk.finish_reason = "timeout_error";

  const Json::Value wrapper = parseJson(tt::dynamo::encode_stream_chunk(chunk));

  ASSERT_TRUE(wrapper.isObject());
  EXPECT_FALSE(wrapper["complete_final"].asBool());
  ASSERT_TRUE(wrapper["data"].isObject());

  const Json::Value& annotated = wrapper["data"];
  EXPECT_TRUE(annotated["data"].isNull())
      << "Dynamo should receive timeout_error as a backend error event";
  EXPECT_EQ(annotated["event"].asString(), "error");
  ASSERT_TRUE(annotated["comment"].isArray());
  ASSERT_EQ(annotated["comment"].size(), 1u);

  const Json::Value errorPayload =
      parseJsonString(annotated["comment"][0].asString());
  EXPECT_EQ(errorPayload["message"].asString(), "timeout_error");
  EXPECT_EQ(errorPayload["code"].asInt(), 500);
}

TEST(DynamoProtocolTest, FinalDataChunkCanCarryStopFinishReason) {
  tt::dynamo::TokenChunk chunk;
  chunk.token_ids = {123};
  chunk.finish_reason = "stop";

  const Json::Value wrapper = parseJson(tt::dynamo::encode_stream_chunk(chunk));

  ASSERT_TRUE(wrapper["data"].isObject());
  ASSERT_TRUE(wrapper["data"]["data"].isObject());
  EXPECT_EQ(wrapper["data"]["data"]["finish_reason"].asString(), "stop");
  ASSERT_TRUE(wrapper["data"]["data"]["token_ids"].isArray());
  EXPECT_EQ(wrapper["data"]["data"]["token_ids"][0].asInt(), 123);
}
