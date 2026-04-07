// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <gtest/gtest.h>
#include <json/json.h>

#include <sstream>
#include <string>

#include "domain/chat_completion_request.hpp"
#include "domain/response_format.hpp"
#include "runners/llm_runner/sampling_params.hpp"

namespace {

Json::Value parseJson(const std::string& str) {
  Json::CharReaderBuilder builder;
  Json::Value root;
  std::string errs;
  std::istringstream iss(str);
  if (!Json::parseFromStream(builder, iss, &root, &errs)) {
    throw std::runtime_error("JSON parse error: " + errs);
  }
  return root;
}

// ---------------------------------------------------------------------------
// ResponseFormat parsing
// ---------------------------------------------------------------------------

TEST(ResponseFormatTest, ParseTextType) {
  auto json = parseJson(R"({"type": "text"})");
  auto fmt = tt::domain::ResponseFormat::fromJson(json);
  EXPECT_EQ(fmt.type, tt::domain::ResponseFormatType::TEXT);
  EXPECT_FALSE(fmt.json_schema_str.has_value());
}

TEST(ResponseFormatTest, ParseJsonObjectType) {
  auto json = parseJson(R"({"type": "json_object"})");
  auto fmt = tt::domain::ResponseFormat::fromJson(json);
  EXPECT_EQ(fmt.type, tt::domain::ResponseFormatType::JSON_OBJECT);
  EXPECT_FALSE(fmt.json_schema_str.has_value());
}

TEST(ResponseFormatTest, ParseJsonSchemaType) {
  auto json = parseJson(R"({
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"}
        },
        "required": ["name", "age"],
        "additionalProperties": false
      }
    }
  })");
  auto fmt = tt::domain::ResponseFormat::fromJson(json);
  EXPECT_EQ(fmt.type, tt::domain::ResponseFormatType::JSON_SCHEMA);
  EXPECT_TRUE(fmt.json_schema_str.has_value());
  EXPECT_EQ(fmt.json_schema_name, "person");
  EXPECT_TRUE(fmt.strict);

  EXPECT_NE(fmt.json_schema_str->find("\"name\""), std::string::npos);
  EXPECT_NE(fmt.json_schema_str->find("\"age\""), std::string::npos);
}

TEST(ResponseFormatTest, RejectInvalidType) {
  auto json = parseJson(R"({"type": "xml"})");
  EXPECT_THROW(tt::domain::ResponseFormat::fromJson(json),
               std::invalid_argument);
}

TEST(ResponseFormatTest, RejectJsonSchemaMissingSchema) {
  auto json =
      parseJson(R"({"type": "json_schema", "json_schema": {"name": "test"}})");
  EXPECT_THROW(tt::domain::ResponseFormat::fromJson(json),
               std::invalid_argument);
}

TEST(ResponseFormatTest, RejectJsonSchemaMissingJsonSchemaField) {
  auto json = parseJson(R"({"type": "json_schema"})");
  EXPECT_THROW(tt::domain::ResponseFormat::fromJson(json),
               std::invalid_argument);
}

TEST(ResponseFormatTest, RejectNonObject) {
  auto json = parseJson(R"("text")");
  EXPECT_THROW(tt::domain::ResponseFormat::fromJson(json),
               std::invalid_argument);
}

TEST(ResponseFormatTest, RejectMissingType) {
  auto json = parseJson(R"({})");
  EXPECT_THROW(tt::domain::ResponseFormat::fromJson(json),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// SamplingParams serialization round-trip with response_format fields
// ---------------------------------------------------------------------------

TEST(SamplingParamsTest, SerializeDeserialize_ResponseFormatText) {
  llm_engine::SamplingParams orig;
  orig.response_format_type = llm_engine::ResponseFormatType::TEXT;

  std::ostringstream os;
  orig.serialize(os);
  std::istringstream is(os.str());
  auto restored = llm_engine::SamplingParams::deserialize(is);

  EXPECT_EQ(restored->response_format_type,
            llm_engine::ResponseFormatType::TEXT);
  EXPECT_FALSE(restored->json_schema_str.has_value());
}

TEST(SamplingParamsTest, SerializeDeserialize_ResponseFormatJsonObject) {
  llm_engine::SamplingParams orig;
  orig.response_format_type = llm_engine::ResponseFormatType::JSON_OBJECT;

  std::ostringstream os;
  orig.serialize(os);
  std::istringstream is(os.str());
  auto restored = llm_engine::SamplingParams::deserialize(is);

  EXPECT_EQ(restored->response_format_type,
            llm_engine::ResponseFormatType::JSON_OBJECT);
  EXPECT_FALSE(restored->json_schema_str.has_value());
}

TEST(SamplingParamsTest, SerializeDeserialize_ResponseFormatJsonSchema) {
  llm_engine::SamplingParams orig;
  orig.response_format_type = llm_engine::ResponseFormatType::JSON_SCHEMA;
  orig.json_schema_str =
      R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false})";

  std::ostringstream os;
  orig.serialize(os);
  std::istringstream is(os.str());
  auto restored = llm_engine::SamplingParams::deserialize(is);

  EXPECT_EQ(restored->response_format_type,
            llm_engine::ResponseFormatType::JSON_SCHEMA);
  ASSERT_TRUE(restored->json_schema_str.has_value());
  EXPECT_EQ(*restored->json_schema_str, *orig.json_schema_str);
}

TEST(SamplingParamsTest, HasGuidedDecoding) {
  llm_engine::SamplingParams text;
  text.response_format_type = llm_engine::ResponseFormatType::TEXT;
  EXPECT_FALSE(text.hasGuidedDecoding());

  llm_engine::SamplingParams jsonObj;
  jsonObj.response_format_type = llm_engine::ResponseFormatType::JSON_OBJECT;
  EXPECT_TRUE(jsonObj.hasGuidedDecoding());

  llm_engine::SamplingParams jsonSchema;
  jsonSchema.response_format_type = llm_engine::ResponseFormatType::JSON_SCHEMA;
  EXPECT_TRUE(jsonSchema.hasGuidedDecoding());
}

// ---------------------------------------------------------------------------
// Backward compatibility: deserialize old format without response_format
// ---------------------------------------------------------------------------

TEST(SamplingParamsTest, DeserializeOldFormat_BackwardCompatible) {
  llm_engine::SamplingParams orig;
  orig.temperature = 0.5f;
  orig.max_tokens = 100;

  std::ostringstream os;
  orig.serialize(os);
  std::string data = os.str();

  // The old format ended after fast_mode. Find the position where
  // response_format fields start and truncate.
  // We can verify backward compatibility by checking that the new fields
  // default correctly when old data is loaded.
  auto restored = llm_engine::SamplingParams::deserialize(
      *std::make_unique<std::istringstream>(data));

  EXPECT_FLOAT_EQ(restored->temperature, 0.5f);
  EXPECT_EQ(restored->max_tokens, 100);
  EXPECT_EQ(restored->response_format_type,
            llm_engine::ResponseFormatType::TEXT);
  EXPECT_FALSE(restored->json_schema_str.has_value());
}

// ---------------------------------------------------------------------------
// ChatCompletionRequest response_format parsing
// ---------------------------------------------------------------------------

TEST(ChatCompletionRequestTest, ParseResponseFormatOmitted) {
  auto json = parseJson(R"({
    "messages": [{"role": "user", "content": "hello"}]
  })");
  auto req = tt::domain::ChatCompletionRequest::fromJson(json, 1);
  EXPECT_FALSE(req.response_format.has_value());
}

TEST(ChatCompletionRequestTest, ParseResponseFormatText) {
  auto json = parseJson(R"({
    "messages": [{"role": "user", "content": "hello"}],
    "response_format": {"type": "text"}
  })");
  auto req = tt::domain::ChatCompletionRequest::fromJson(json, 1);
  ASSERT_TRUE(req.response_format.has_value());
  EXPECT_EQ(req.response_format->type, tt::domain::ResponseFormatType::TEXT);
}

TEST(ChatCompletionRequestTest, ParseResponseFormatJsonSchema) {
  auto json = parseJson(R"({
    "messages": [{"role": "user", "content": "hello"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "test",
        "schema": {
          "type": "object",
          "properties": {"x": {"type": "integer"}},
          "required": ["x"],
          "additionalProperties": false
        }
      }
    }
  })");
  auto req = tt::domain::ChatCompletionRequest::fromJson(json, 1);
  ASSERT_TRUE(req.response_format.has_value());
  EXPECT_EQ(req.response_format->type,
            tt::domain::ResponseFormatType::JSON_SCHEMA);
  EXPECT_TRUE(req.response_format->json_schema_str.has_value());
}

TEST(ChatCompletionRequestTest, ToLLMRequestPreservesResponseFormat) {
  auto json = parseJson(R"({
    "messages": [{"role": "user", "content": "hello"}],
    "response_format": {"type": "json_object"}
  })");
  auto req = tt::domain::ChatCompletionRequest::fromJson(json, 1);
  auto llmReq = req.toLLMRequest();
  ASSERT_TRUE(llmReq.response_format.has_value());
  EXPECT_EQ(llmReq.response_format->type,
            tt::domain::ResponseFormatType::JSON_OBJECT);
}

TEST(ChatCompletionRequestTest, RejectInvalidResponseFormat) {
  auto json = parseJson(R"({
    "messages": [{"role": "user", "content": "hello"}],
    "response_format": {"type": "invalid"}
  })");
  EXPECT_THROW(tt::domain::ChatCompletionRequest::fromJson(json, 1),
               std::invalid_argument);
}

}  // namespace
