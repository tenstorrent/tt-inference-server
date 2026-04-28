// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>
#include <json/json.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "domain/chat_completion_request.hpp"
#include "domain/response_format.hpp"
#include "domain/sampling_params.hpp"
#include "runners/guided_decoder_manager.hpp"
#include "utils/tokenizers/tokenizer.hpp"

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
  tt::domain::SamplingParams orig;
  orig.response_format_type = tt::domain::ResponseFormatType::TEXT;

  std::ostringstream os;
  orig.serialize(os);
  std::istringstream is(os.str());
  auto restored = tt::domain::SamplingParams::deserialize(is);

  EXPECT_EQ(restored->response_format_type,
            tt::domain::ResponseFormatType::TEXT);
  EXPECT_FALSE(restored->json_schema_str.has_value());
}

TEST(SamplingParamsTest, SerializeDeserialize_ResponseFormatJsonObject) {
  tt::domain::SamplingParams orig;
  orig.response_format_type = tt::domain::ResponseFormatType::JSON_OBJECT;

  std::ostringstream os;
  orig.serialize(os);
  std::istringstream is(os.str());
  auto restored = tt::domain::SamplingParams::deserialize(is);

  EXPECT_EQ(restored->response_format_type,
            tt::domain::ResponseFormatType::JSON_OBJECT);
  EXPECT_FALSE(restored->json_schema_str.has_value());
}

TEST(SamplingParamsTest, SerializeDeserialize_ResponseFormatJsonSchema) {
  tt::domain::SamplingParams orig;
  orig.response_format_type = tt::domain::ResponseFormatType::JSON_SCHEMA;
  orig.json_schema_str =
      R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false})";

  std::ostringstream os;
  orig.serialize(os);
  std::istringstream is(os.str());
  auto restored = tt::domain::SamplingParams::deserialize(is);

  EXPECT_EQ(restored->response_format_type,
            tt::domain::ResponseFormatType::JSON_SCHEMA);
  ASSERT_TRUE(restored->json_schema_str.has_value());
  EXPECT_EQ(*restored->json_schema_str, *orig.json_schema_str);
}

TEST(SamplingParamsTest, HasGuidedDecoding) {
  tt::domain::SamplingParams text;
  text.response_format_type = tt::domain::ResponseFormatType::TEXT;
  EXPECT_FALSE(text.hasGuidedDecoding());

  tt::domain::SamplingParams jsonObj;
  jsonObj.response_format_type = tt::domain::ResponseFormatType::JSON_OBJECT;
  EXPECT_TRUE(jsonObj.hasGuidedDecoding());

  tt::domain::SamplingParams jsonSchema;
  jsonSchema.response_format_type = tt::domain::ResponseFormatType::JSON_SCHEMA;
  EXPECT_TRUE(jsonSchema.hasGuidedDecoding());
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

// GuidedDecoderManager — bitmask, token acceptance, and grammar completion.
// DeepSeek-R1-0528 single-character token IDs verified against tokenizer.json.
static constexpr int K_JSON_OPEN_BRACE  = 93;  // {
static constexpr int K_JSON_CLOSE_BRACE = 95;  // }
static constexpr int K_JSON_QUOTE       = 4;   // "
static constexpr int K_JSON_COLON       = 28;  // :
static constexpr int K_JSON_DIGIT_4     = 22;  // '4'
static constexpr int K_JSON_LETTER_x    = 90;  // 'x'
static constexpr int K_JSON_LETTER_A    = 35;  // 'A' – invalid outside strings

class GuidedDecoderManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto& tok = tt::utils::tokenizers::activeTokenizer();
    vocab_     = tok.getEncodedVocab();
    vocabSize_ = static_cast<int>(vocab_.size());
    for (int64_t id : tok.stopTokenIds()) {
      stopIds_.push_back(static_cast<int32_t>(id));
    }
    ASSERT_FALSE(stopIds_.empty()) << "Tokenizer must expose at least one EOS token";
    decoder_ = std::make_unique<tt::runners::GuidedDecoderManager>(
        vocab_, vocabSize_, stopIds_);
  }

  // Schema: {"x": integer} — minimal, deterministic, fast to compile.
  static tt::domain::SamplingParams integerXSchema() {
    tt::domain::SamplingParams sp;
    sp.response_format_type = tt::domain::ResponseFormatType::JSON_SCHEMA;
    sp.json_schema_str =
        R"({"type":"object","properties":{"x":{"type":"integer"}})"
        R"(,"required":["x"],"additionalProperties":false})";
    return sp;
  }

  static bool isBitmaskSet(const std::vector<int32_t>& mask, int tokenId) {
    if (tokenId < 0) return false;
    size_t word = static_cast<size_t>(tokenId) / 32;
    if (word >= mask.size()) return false;
    return (static_cast<uint32_t>(mask[word]) >> (tokenId % 32)) & 1;
  }

  std::vector<std::string> vocab_;
  int vocabSize_ = 0;
  std::vector<int32_t> stopIds_;
  std::unique_ptr<tt::runners::GuidedDecoderManager> decoder_;
};

// The very first bitmask for a JSON-schema request must allow '{' and must
// not allow an uppercase letter, which is only valid inside string values.
TEST_F(GuidedDecoderManagerTest, InitialBitmaskAllowsOpenBrace) {
  decoder_->initRequest(1, integerXSchema());

  std::vector<int32_t> bitmask;
  decoder_->fillNextBitmask(1, bitmask);

  EXPECT_FALSE(bitmask.empty());
  EXPECT_TRUE(isBitmaskSet(bitmask, K_JSON_OPEN_BRACE))
      << "'{' must be valid at the start of a JSON schema response";
  EXPECT_FALSE(isBitmaskSet(bitmask, K_JSON_LETTER_A))
      << "'A' must not be valid at the start of a JSON schema response";
}

// Feeding the exact token sequence for {"x":4} followed by the EOS token
// must be fully accepted and mark the grammar as complete only after EOS.
TEST_F(GuidedDecoderManagerTest, AcceptsValidJsonSequenceAndCompletesOnEos) {
  decoder_->initRequest(1, integerXSchema());

  const int32_t tokens[] = {K_JSON_OPEN_BRACE, K_JSON_QUOTE, K_JSON_LETTER_x,
                             K_JSON_QUOTE,      K_JSON_COLON, K_JSON_DIGIT_4,
                             K_JSON_CLOSE_BRACE};
  for (int32_t tid : tokens) {
    auto r = decoder_->acceptToken(1, tid);
    EXPECT_TRUE(r.accepted)    << "Token " << tid << " should be accepted";
    EXPECT_FALSE(r.completed)  << "Grammar must not complete before EOS";
  }

  // EOS triggers IsTerminated() → completed = true.
  auto r = decoder_->acceptToken(1, stopIds_.front());
  EXPECT_TRUE(r.accepted);
  EXPECT_TRUE(r.completed) << "Grammar must complete after EOS token";
}

// Presenting an uppercase letter as the first token must be rejected because
// the grammar expects '{' (or whitespace), not a string character.
TEST_F(GuidedDecoderManagerTest, RejectsTokenOutsideGrammar) {
  decoder_->initRequest(1, integerXSchema());

  auto r = decoder_->acceptToken(1, K_JSON_LETTER_A);
  EXPECT_FALSE(r.accepted) << "'A' must be rejected when grammar expects '{'";
}
