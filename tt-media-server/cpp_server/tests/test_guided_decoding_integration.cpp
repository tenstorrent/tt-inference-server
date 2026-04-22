// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include <gtest/gtest.h>
#include <json/json.h>

#include "domain/chat_completion_request.hpp"
#include "domain/sampling_params.hpp"
#include "utils/mapper.hpp"

using namespace tt::domain;
using namespace tt::domain::tool_calls;

class GuidedDecodingIntegrationTest : public ::testing::Test {
 protected:
  Json::Value createWeatherFunction() {
    Json::Value func;
    func["name"] = "get_weather";
    func["description"] = "Get weather for a location";

    Json::Value params;
    params["type"] = "object";
    params["properties"]["location"]["type"] = "string";
    params["properties"]["location"]["description"] = "City name";
    params["properties"]["unit"]["type"] = "string";
    params["properties"]["unit"]["enum"].append("celsius");
    params["properties"]["unit"]["enum"].append("fahrenheit");
    params["required"].append("location");
    func["parameters"] = params;

    return func;
  }

  Json::Value createRequest(const Json::Value& toolChoice) {
    Json::Value json;
    json["model"] = "test-model";

    Json::Value msg;
    msg["role"] = "user";
    msg["content"] = "What's the weather?";
    json["messages"].append(msg);

    Json::Value tool;
    tool["type"] = "function";
    tool["function"] = createWeatherFunction();
    json["tools"].append(tool);

    json["tool_choice"] = toolChoice;

    return json;
  }
};

TEST_F(GuidedDecodingIntegrationTest,
       FunctionToolChoiceCreatesGuidedDecoding) {
  // Create request with tool_choice = function
  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"]["name"] = "get_weather";

  Json::Value requestJson = createRequest(toolChoice);
  auto chatRequest = ChatCompletionRequest::fromJson(requestJson, 1);
  auto llmRequest = chatRequest.toLLMRequest();

  // Verify response_format is set for guided decoding
  ASSERT_TRUE(llmRequest.response_format.has_value())
      << "response_format should be set for tool_choice=function";

  auto& format = llmRequest.response_format.value();

  EXPECT_EQ(format.type, tt::config::ResponseFormatType::JSON_SCHEMA)
      << "Should use JSON_SCHEMA for guided decoding";

  EXPECT_TRUE(format.strict) << "Should enforce strict schema adherence";

  EXPECT_TRUE(format.json_schema_str.has_value())
      << "JSON schema string should be present";

  // Verify the schema contains the function parameters
  std::string schema = format.json_schema_str.value();
  EXPECT_TRUE(schema.find("location") != std::string::npos)
      << "Schema should contain 'location' parameter";
  EXPECT_TRUE(schema.find("string") != std::string::npos)
      << "Schema should specify string type";
  EXPECT_TRUE(schema.find("unit") != std::string::npos)
      << "Schema should contain 'unit' parameter";
}

TEST_F(GuidedDecodingIntegrationTest, SamplingParamsMapsGuidedDecoding) {
  // Create request with tool_choice = function
  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"]["name"] = "get_weather";

  Json::Value requestJson = createRequest(toolChoice);
  auto chatRequest = ChatCompletionRequest::fromJson(requestJson, 1);
  auto llmRequest = chatRequest.toLLMRequest();

  // Map to SamplingParams (this is what the LLM runner uses)
  auto samplingParams = tt::utils::mapper::mapSamplingParams(llmRequest);

  // Verify guided decoding is enabled
  EXPECT_TRUE(samplingParams.hasGuidedDecoding())
      << "SamplingParams should indicate guided decoding is active";

  EXPECT_EQ(samplingParams.response_format_type,
            tt::config::ResponseFormatType::JSON_SCHEMA)
      << "Response format type should be JSON_SCHEMA";

  ASSERT_TRUE(samplingParams.json_schema_str.has_value())
      << "JSON schema string should be mapped to SamplingParams";

  // Verify the schema matches what we expect
  std::string schema = samplingParams.json_schema_str.value();
  EXPECT_TRUE(schema.find("location") != std::string::npos);
  EXPECT_TRUE(schema.find("unit") != std::string::npos);
  EXPECT_TRUE(schema.find("required") != std::string::npos);
}

TEST_F(GuidedDecodingIntegrationTest,
       AutoToolChoiceDoesNotEnableGuidedDecoding) {
  // Create request with tool_choice = auto
  Json::Value toolChoice = "auto";

  Json::Value requestJson = createRequest(toolChoice);
  auto chatRequest = ChatCompletionRequest::fromJson(requestJson, 1);
  auto llmRequest = chatRequest.toLLMRequest();

  // Map to SamplingParams
  auto samplingParams = tt::utils::mapper::mapSamplingParams(llmRequest);

  // Verify guided decoding is NOT enabled for auto
  EXPECT_FALSE(samplingParams.hasGuidedDecoding())
      << "Auto mode should not use guided decoding";

  EXPECT_EQ(samplingParams.response_format_type,
            tt::config::ResponseFormatType::TEXT)
      << "Auto mode should use TEXT format";
}

TEST_F(GuidedDecodingIntegrationTest, NoneToolChoiceDoesNotEnableGuidedDecoding) {
  // Create request with tool_choice = none
  Json::Value toolChoice = "none";

  Json::Value requestJson = createRequest(toolChoice);
  auto chatRequest = ChatCompletionRequest::fromJson(requestJson, 1);
  auto llmRequest = chatRequest.toLLMRequest();

  // Map to SamplingParams
  auto samplingParams = tt::utils::mapper::mapSamplingParams(llmRequest);

  // Verify guided decoding is NOT enabled for none
  EXPECT_FALSE(samplingParams.hasGuidedDecoding())
      << "None mode should not use guided decoding";
}

TEST_F(GuidedDecodingIntegrationTest, SchemaContainsEnumConstraints) {
  // Create request
  Json::Value toolChoice;
  toolChoice["type"] = "function";
  toolChoice["function"]["name"] = "get_weather";

  Json::Value requestJson = createRequest(toolChoice);
  auto chatRequest = ChatCompletionRequest::fromJson(requestJson, 1);
  auto llmRequest = chatRequest.toLLMRequest();

  ASSERT_TRUE(llmRequest.response_format.has_value());
  std::string schema = llmRequest.response_format->json_schema_str.value();

  // Verify enum constraints are preserved
  EXPECT_TRUE(schema.find("celsius") != std::string::npos)
      << "Schema should contain enum value 'celsius'";
  EXPECT_TRUE(schema.find("fahrenheit") != std::string::npos)
      << "Schema should contain enum value 'fahrenheit'";
  EXPECT_TRUE(schema.find("enum") != std::string::npos)
      << "Schema should specify enum constraint";
}

// Main function for running tests
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
