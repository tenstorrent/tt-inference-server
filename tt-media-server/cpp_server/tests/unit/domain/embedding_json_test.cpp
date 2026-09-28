// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Unit tests for the embedding domain JSON contracts:
// EmbeddingRequest fromJson/toJson (HTTP body + worker IPC) and
// EmbeddingResponse toOpenaiJson/fromJson (the external API schema).

#include <gtest/gtest.h>
#include <json/json.h>

#include <stdexcept>
#include <vector>

#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"

using tt::domain::EmbeddingRequest;
using tt::domain::EmbeddingResponse;

namespace {

Json::Value parseJson(const std::string& text) {
  Json::Value root;
  Json::CharReaderBuilder builder;
  std::string errors;
  std::istringstream iss(text);
  if (!Json::parseFromStream(builder, iss, &root, &errors)) {
    ADD_FAILURE() << "test JSON did not parse: " << errors;
  }
  return root;
}

}  // namespace

// All fields present parse into the request; task_id comes from the caller.
TEST(EmbeddingRequestJsonTest, FromJsonParsesAllFields) {
  const auto json = parseJson(
      R"({"model":"BAAI/bge-m3","input":"embed me","user":"user-7"})");

  const auto req = EmbeddingRequest::fromJson(json, 42);

  EXPECT_EQ(req.task_id, 42u);
  EXPECT_EQ(req.model, "BAAI/bge-m3");
  EXPECT_EQ(req.input, "embed me");
  ASSERT_TRUE(req.user.has_value());
  EXPECT_EQ(*req.user, "user-7");
}

// Absent fields default to empty/nullopt instead of failing; the controller
// checks the required "input" itself and fills the default model.
TEST(EmbeddingRequestJsonTest, FromJsonDefaultsMissingFields) {
  const auto req = EmbeddingRequest::fromJson(parseJson("{}"), 1);

  EXPECT_EQ(req.task_id, 1u);
  EXPECT_TRUE(req.model.empty());
  EXPECT_TRUE(req.input.empty());
  EXPECT_FALSE(req.user.has_value());
}

// Wrong-typed fields throw invalid_argument, which the controller maps to
// HTTP 400.
TEST(EmbeddingRequestJsonTest, FromJsonThrowsOnWrongTypes) {
  EXPECT_THROW(EmbeddingRequest::fromJson(parseJson(R"({"input":123})"), 1),
               std::invalid_argument);
  EXPECT_THROW(EmbeddingRequest::fromJson(parseJson(R"({"model":["a"]})"), 1),
               std::invalid_argument);
  EXPECT_THROW(EmbeddingRequest::fromJson(parseJson(R"({"user":false})"), 1),
               std::invalid_argument);
}

// toJson -> fromJson round-trips every field; this is the parent->worker IPC
// path (encodeBatchJson / parseBatch).
TEST(EmbeddingRequestJsonTest, ToJsonFromJsonRoundTrip) {
  EmbeddingRequest req(99);
  req.model = "BAAI/bge-large-en-v1.5";
  req.input = "text with \"quotes\" and\nnewlines";
  req.user = "someone";

  const auto json = req.toJson();
  const auto back = EmbeddingRequest::fromJson(json, json["task_id"].asUInt());

  EXPECT_EQ(back.task_id, 99u);
  EXPECT_EQ(back.model, req.model);
  EXPECT_EQ(back.input, req.input);
  ASSERT_TRUE(back.user.has_value());
  EXPECT_EQ(*back.user, "someone");
}

// The optional user field is omitted from the wire when unset.
TEST(EmbeddingRequestJsonTest, ToJsonOmitsUnsetUser) {
  EmbeddingRequest req(5);
  req.model = "m";
  req.input = "i";

  EXPECT_FALSE(req.toJson().isMember("user"));
}

// The OpenAI response schema: exact field names and structure clients
// parse by name.
TEST(EmbeddingResponseJsonTest, ToOpenaiJsonMatchesSchema) {
  EmbeddingResponse resp(7);
  resp.embedding = {0.5f, -1.25f, 0.125f};
  resp.total_tokens = 12;
  resp.model = "BAAI/bge-large-en-v1.5";

  const auto json = resp.toOpenaiJson();

  EXPECT_EQ(json["object"].asString(), "list");
  EXPECT_EQ(json["model"].asString(), "BAAI/bge-large-en-v1.5");
  ASSERT_TRUE(json["data"].isArray());
  ASSERT_EQ(json["data"].size(), 1u);
  const auto& entry = json["data"][0];
  EXPECT_EQ(entry["object"].asString(), "embedding");
  EXPECT_EQ(entry["index"].asInt(), 0);
  ASSERT_EQ(entry["embedding"].size(), 3u);
  EXPECT_EQ(entry["embedding"][0].asFloat(), 0.5f);
  EXPECT_EQ(entry["embedding"][1].asFloat(), -1.25f);
  EXPECT_EQ(entry["embedding"][2].asFloat(), 0.125f);
  EXPECT_EQ(json["usage"]["total_tokens"].asInt(), 12);
  EXPECT_EQ(json["usage"]["prompt_tokens"].asInt(), 12);
}

// Float values survive the float->double->float JSON conversion exactly.
TEST(EmbeddingResponseJsonTest, ToOpenaiJsonPreservesFloatValues) {
  EmbeddingResponse resp(1);
  for (int i = 0; i < 64; ++i) {
    resp.embedding.push_back(static_cast<float>(i - 32) / 7.0f);
  }

  const auto json = resp.toOpenaiJson();
  const auto& arr = json["data"][0]["embedding"];

  ASSERT_EQ(arr.size(), resp.embedding.size());
  for (Json::ArrayIndex i = 0; i < arr.size(); ++i) {
    EXPECT_EQ(arr[i].asFloat(), resp.embedding[i]) << "index " << i;
  }
}

// An empty embedding still produces the full schema with an empty array.
TEST(EmbeddingResponseJsonTest, ToOpenaiJsonWithEmptyEmbedding) {
  EmbeddingResponse resp(3);
  resp.model = "m";

  const auto json = resp.toOpenaiJson();

  ASSERT_EQ(json["data"].size(), 1u);
  EXPECT_TRUE(json["data"][0]["embedding"].isArray());
  EXPECT_EQ(json["data"][0]["embedding"].size(), 0u);
}

// fromJson parses all fields including the error.
TEST(EmbeddingResponseJsonTest, FromJsonParsesAllFields) {
  const auto json = parseJson(
      R"({"task_id":11,"embedding":[1.0,2.0],"total_tokens":5,)"
      R"("model":"m","error":"boom"})");

  const auto resp = EmbeddingResponse::fromJson(json);

  EXPECT_EQ(resp.task_id, 11u);
  ASSERT_EQ(resp.embedding.size(), 2u);
  EXPECT_EQ(resp.embedding[0], 1.0f);
  EXPECT_EQ(resp.embedding[1], 2.0f);
  EXPECT_EQ(resp.total_tokens, 5);
  EXPECT_EQ(resp.model, "m");
  EXPECT_EQ(resp.error, "boom");
}

// Missing fields default sanely; a missing task_id falls back to a
// generated one (unique across parses).
TEST(EmbeddingResponseJsonTest, FromJsonDefaultsMissingFields) {
  const auto first = EmbeddingResponse::fromJson(parseJson("{}"));
  const auto second = EmbeddingResponse::fromJson(parseJson("{}"));

  EXPECT_TRUE(first.embedding.empty());
  EXPECT_EQ(first.total_tokens, 0);
  EXPECT_TRUE(first.model.empty());
  EXPECT_TRUE(first.error.empty());
  EXPECT_NE(first.task_id, second.task_id);
}
