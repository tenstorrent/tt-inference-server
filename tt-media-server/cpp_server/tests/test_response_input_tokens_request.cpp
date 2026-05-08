// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <gtest/gtest.h>
#include <json/json.h>

#include <stdexcept>
#include <string>

#include "domain/llm/response_input_tokens_request.hpp"

using tt::domain::llm::ResponseInputTokensRequest;

namespace {

constexpr uint32_t TASK_ID = 42;

Json::Value parseJson(const std::string& text) {
  Json::CharReaderBuilder b;
  Json::Value v;
  std::string err;
  std::istringstream in(text);
  EXPECT_TRUE(Json::parseFromStream(b, in, &v, &err)) << err;
  return v;
}

}  // namespace

// Per OpenAI spec, all body fields are optional: empty body must parse cleanly.
TEST(ResponseInputTokensRequest, EmptyBodyAcceptedPerSpec) {
  Json::Value body(Json::objectValue);
  auto req = ResponseInputTokensRequest::fromJson(body, TASK_ID);
  EXPECT_TRUE(req.input.isNull());
  EXPECT_TRUE(req.conversation.isNull());
  EXPECT_FALSE(req.instructions.has_value());
  EXPECT_FALSE(req.model.has_value());
  EXPECT_FALSE(req.truncation.has_value());
}

TEST(ResponseInputTokensRequest, AllSpecFieldsParsed) {
  auto body = parseJson(R"({
    "conversation": {"id": "conv_123"},
    "input": "Tell me a joke.",
    "instructions": "Be brief.",
    "model": "gpt-5",
    "parallel_tool_calls": true,
    "reasoning": {"effort": "low"},
    "text": {"format": {"type": "text"}, "verbosity": "low"},
    "tool_choice": "auto",
    "tools": [{"type": "function", "name": "get_weather"}],
    "truncation": "auto"
  })");

  auto req = ResponseInputTokensRequest::fromJson(body, TASK_ID);

  EXPECT_TRUE(req.conversation.isObject());
  EXPECT_EQ(req.conversation["id"].asString(), "conv_123");
  EXPECT_EQ(req.input.asString(), "Tell me a joke.");
  ASSERT_TRUE(req.instructions.has_value());
  EXPECT_EQ(*req.instructions, "Be brief.");
  ASSERT_TRUE(req.model.has_value());
  EXPECT_EQ(*req.model, "gpt-5");
  ASSERT_TRUE(req.parallel_tool_calls.has_value());
  EXPECT_TRUE(*req.parallel_tool_calls);
  EXPECT_EQ(req.reasoning["effort"].asString(), "low");
  EXPECT_EQ(req.text["verbosity"].asString(), "low");
  EXPECT_EQ(req.tool_choice.asString(), "auto");
  ASSERT_TRUE(req.tools.isArray());
  EXPECT_EQ(req.tools.size(), 1u);
  EXPECT_EQ(req.tools[0]["type"].asString(), "function");
  ASSERT_TRUE(req.truncation.has_value());
  EXPECT_EQ(*req.truncation, "auto");
}

TEST(ResponseInputTokensRequest, ConversationStringForm) {
  auto body = parseJson(R"({"conversation": "conv_abc"})");
  auto req = ResponseInputTokensRequest::fromJson(body, TASK_ID);
  EXPECT_TRUE(req.conversation.isString());
  EXPECT_EQ(req.conversation.asString(), "conv_abc");
}

TEST(ResponseInputTokensRequest, ConversationObjectMissingIdRejected) {
  auto body = parseJson(R"({"conversation": {}})");
  EXPECT_THROW(ResponseInputTokensRequest::fromJson(body, TASK_ID),
               std::invalid_argument);
}

TEST(ResponseInputTokensRequest, ConversationWrongTypeRejected) {
  auto body = parseJson(R"({"conversation": 42})");
  EXPECT_THROW(ResponseInputTokensRequest::fromJson(body, TASK_ID),
               std::invalid_argument);
}

TEST(ResponseInputTokensRequest, InputArrayAccepted) {
  auto body = parseJson(R"({"input": [{"role": "user", "content": "hi"}]})");
  auto req = ResponseInputTokensRequest::fromJson(body, TASK_ID);
  ASSERT_TRUE(req.input.isArray());
  EXPECT_EQ(req.input.size(), 1u);
}

TEST(ResponseInputTokensRequest, InputWrongTypeRejected) {
  auto body = parseJson(R"({"input": 42})");
  EXPECT_THROW(ResponseInputTokensRequest::fromJson(body, TASK_ID),
               std::invalid_argument);
}

TEST(ResponseInputTokensRequest, TruncationEnumValidated) {
  auto bad = parseJson(R"({"truncation": "sometimes"})");
  EXPECT_THROW(ResponseInputTokensRequest::fromJson(bad, TASK_ID),
               std::invalid_argument);

  auto okAuto = parseJson(R"({"truncation": "auto"})");
  EXPECT_NO_THROW(ResponseInputTokensRequest::fromJson(okAuto, TASK_ID));

  auto okDisabled = parseJson(R"({"truncation": "disabled"})");
  EXPECT_NO_THROW(ResponseInputTokensRequest::fromJson(okDisabled, TASK_ID));
}

TEST(ResponseInputTokensRequest,
     PreviousResponseIdAndConversationMutuallyExclusive) {
  auto body = parseJson(R"({
    "conversation": "conv_1",
    "previous_response_id": "resp_1"
  })");
  EXPECT_THROW(ResponseInputTokensRequest::fromJson(body, TASK_ID),
               std::invalid_argument);
}

TEST(ResponseInputTokensRequest, PreviousResponseIdAloneAccepted) {
  auto body = parseJson(R"({"previous_response_id": "resp_1"})");
  auto req = ResponseInputTokensRequest::fromJson(body, TASK_ID);
  ASSERT_TRUE(req.previous_response_id.has_value());
  EXPECT_EQ(*req.previous_response_id, "resp_1");
}

TEST(ResponseInputTokensRequest, ReasoningMustBeObject) {
  auto body = parseJson(R"({"reasoning": "low"})");
  EXPECT_THROW(ResponseInputTokensRequest::fromJson(body, TASK_ID),
               std::invalid_argument);
}

TEST(ResponseInputTokensRequest, ToolsMustBeArray) {
  auto body = parseJson(R"({"tools": {"name": "x"}})");
  EXPECT_THROW(ResponseInputTokensRequest::fromJson(body, TASK_ID),
               std::invalid_argument);
}

// toResponsesRequest must forward token-affecting fields so the count reflects
// what POST /v1/responses would actually send to the model.
TEST(ResponseInputTokensRequest,
     ToResponsesRequestForwardsTokenAffectingFields) {
  auto body = parseJson(R"({
    "input": "hello",
    "instructions": "You are concise.",
    "model": "gpt-5",
    "tools": [{"type": "function", "name": "f"}],
    "tool_choice": "auto",
    "text": {"verbosity": "low"},
    "reasoning": {"effort": "low"},
    "truncation": "auto",
    "parallel_tool_calls": false
  })");
  auto req = ResponseInputTokensRequest::fromJson(body, TASK_ID);
  auto resp = req.toResponsesRequest();

  EXPECT_EQ(resp.input.asString(), "hello");
  ASSERT_TRUE(resp.instructions.has_value());
  EXPECT_EQ(*resp.instructions, "You are concise.");
  ASSERT_TRUE(resp.model.has_value());
  EXPECT_EQ(*resp.model, "gpt-5");
  ASSERT_TRUE(resp.tools.isArray());
  EXPECT_EQ(resp.tools.size(), 1u);
  EXPECT_FALSE(resp.tool_choice.isNull());
  EXPECT_FALSE(resp.text.isNull());
  EXPECT_FALSE(resp.reasoning.isNull());
  ASSERT_TRUE(resp.truncation.has_value());
  EXPECT_EQ(*resp.truncation, "auto");
  ASSERT_TRUE(resp.parallel_tool_calls.has_value());
  EXPECT_FALSE(*resp.parallel_tool_calls);
}

TEST(ResponseInputTokensRequest, TaskIdPreserved) {
  Json::Value body(Json::objectValue);
  auto req = ResponseInputTokensRequest::fromJson(body, TASK_ID);
  EXPECT_EQ(req.task_id, TASK_ID);
  EXPECT_EQ(req.toJson()["task_id"].asUInt(), TASK_ID);
}
