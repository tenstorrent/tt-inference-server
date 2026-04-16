// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "domain/base_response.hpp"
#include "domain/json_field.hpp"
#include "domain/responses_request.hpp"
#include "runners/llm_runner/sampling_params.hpp"
#include "utils/id_generator.hpp"

namespace tt::domain {

struct IncompleteDetails {
  std::string reason;

  Json::Value toJson() const {
    Json::Value j;
    j["reason"] = reason;
    return j;
  }

  static IncompleteDetails fromJson(const Json::Value& json) {
    using namespace json_field;
    checkObject(json, "incomplete_details");
    IncompleteDetails d;
    d.reason = getString(json["reason"], "reason");
    return d;
  }
};

struct InputTokensDetails {
  int cached_tokens = 0;
  std::vector<int> input_tokens_per_turn;
  std::vector<int> cached_tokens_per_turn;

  Json::Value toJson() const {
    Json::Value j;
    j["cached_tokens"] = cached_tokens;
    writeIntArray(j, "input_tokens_per_turn", input_tokens_per_turn);
    writeIntArray(j, "cached_tokens_per_turn", cached_tokens_per_turn);
    return j;
  }

  static InputTokensDetails fromJson(const Json::Value& json) {
    using namespace json_field;
    checkObject(json, "input_tokens_details");
    InputTokensDetails d;
    if (json.isMember("cached_tokens"))
      d.cached_tokens = getInt(json["cached_tokens"], "cached_tokens");
    if (json.isMember("input_tokens_per_turn"))
      d.input_tokens_per_turn =
          readIntArray(json["input_tokens_per_turn"], "input_tokens_per_turn");
    if (json.isMember("cached_tokens_per_turn"))
      d.cached_tokens_per_turn = readIntArray(json["cached_tokens_per_turn"],
                                              "cached_tokens_per_turn");
    return d;
  }

 private:
  static void writeIntArray(Json::Value& parent, const char* key,
                            const std::vector<int>& vec) {
    Json::Value arr(Json::arrayValue);
    for (int v : vec) arr.append(v);
    parent[key] = std::move(arr);
  }

  static std::vector<int> readIntArray(const Json::Value& json,
                                       const char* field) {
    json_field::checkArray(json, field);
    std::vector<int> out;
    for (const auto& v : json) {
      if (!v.isIntegral())
        throw std::invalid_argument(std::string(field) +
                                    " elements must be integers");
      out.push_back(v.asInt());
    }
    return out;
  }
};

struct OutputTokensDetails {
  int reasoning_tokens = 0;
  int tool_output_tokens = 0;
  std::vector<int> output_tokens_per_turn;
  std::vector<int> tool_output_tokens_per_turn;

  Json::Value toJson() const {
    Json::Value j;
    j["reasoning_tokens"] = reasoning_tokens;
    j["tool_output_tokens"] = tool_output_tokens;
    writeIntArray(j, "output_tokens_per_turn", output_tokens_per_turn);
    writeIntArray(j, "tool_output_tokens_per_turn",
                  tool_output_tokens_per_turn);
    return j;
  }

  static OutputTokensDetails fromJson(const Json::Value& json) {
    using namespace json_field;
    checkObject(json, "output_tokens_details");
    OutputTokensDetails d;
    if (json.isMember("reasoning_tokens"))
      d.reasoning_tokens = getInt(json["reasoning_tokens"], "reasoning_tokens");
    if (json.isMember("tool_output_tokens"))
      d.tool_output_tokens =
          getInt(json["tool_output_tokens"], "tool_output_tokens");
    if (json.isMember("output_tokens_per_turn"))
      d.output_tokens_per_turn = readIntArray(
          json["output_tokens_per_turn"], "output_tokens_per_turn");
    if (json.isMember("tool_output_tokens_per_turn"))
      d.tool_output_tokens_per_turn =
          readIntArray(json["tool_output_tokens_per_turn"],
                       "tool_output_tokens_per_turn");
    return d;
  }

 private:
  static void writeIntArray(Json::Value& parent, const char* key,
                            const std::vector<int>& vec) {
    Json::Value arr(Json::arrayValue);
    for (int v : vec) arr.append(v);
    parent[key] = std::move(arr);
  }

  static std::vector<int> readIntArray(const Json::Value& json,
                                       const char* field) {
    json_field::checkArray(json, field);
    std::vector<int> out;
    for (const auto& v : json) {
      if (!v.isIntegral())
        throw std::invalid_argument(std::string(field) +
                                    " elements must be integers");
      out.push_back(v.asInt());
    }
    return out;
  }
};

struct ResponseUsage {
  int input_tokens = 0;
  InputTokensDetails input_tokens_details{};
  int output_tokens = 0;
  OutputTokensDetails output_tokens_details{};
  int total_tokens = 0;

  Json::Value toJson() const {
    Json::Value j;
    j["input_tokens"] = input_tokens;
    j["input_tokens_details"] = input_tokens_details.toJson();
    j["output_tokens"] = output_tokens;
    j["output_tokens_details"] = output_tokens_details.toJson();
    j["total_tokens"] = total_tokens;
    return j;
  }

  static ResponseUsage fromJson(const Json::Value& json) {
    using namespace json_field;
    checkObject(json, "usage");
    ResponseUsage u;
    if (json.isMember("input_tokens"))
      u.input_tokens = getInt(json["input_tokens"], "input_tokens");
    if (json.isMember("input_tokens_details"))
      u.input_tokens_details =
          InputTokensDetails::fromJson(json["input_tokens_details"]);
    if (json.isMember("output_tokens"))
      u.output_tokens = getInt(json["output_tokens"], "output_tokens");
    if (json.isMember("output_tokens_details"))
      u.output_tokens_details =
          OutputTokensDetails::fromJson(json["output_tokens_details"]);
    if (json.isMember("total_tokens"))
      u.total_tokens = getInt(json["total_tokens"], "total_tokens");
    return u;
  }
};

/**
 * OpenAI-compatible Responses API response plus vLLM extensions.
 * Mirrors vllm ResponsesResponse (protocol.py).
 */
struct ResponsesResponse : BaseResponse {
  using BaseResponse::BaseResponse;

  std::string id;
  int64_t created_at = 0;

  std::optional<IncompleteDetails> incomplete_details;
  std::optional<std::string> instructions;
  Json::Value metadata = Json::Value(Json::objectValue);

  std::string model;
  std::string object = "response";
  Json::Value output = Json::Value(Json::arrayValue);

  bool parallel_tool_calls = false;
  float temperature = 0.0f;
  Json::Value tool_choice;
  Json::Value tools = Json::Value(Json::arrayValue);
  float top_p = 0.0f;

  bool background = false;
  int max_output_tokens = 0;
  std::optional<int> max_tool_calls;

  std::optional<std::string> previous_response_id;
  Json::Value prompt;
  Json::Value reasoning;

  std::string service_tier;
  std::string status;

  Json::Value text;
  std::optional<int> top_logprobs;
  std::string truncation;

  std::optional<ResponseUsage> usage;
  std::optional<std::string> user;
  std::optional<float> presence_penalty;
  std::optional<float> frequency_penalty;

  Json::Value kv_transfer_params;
  Json::Value input_messages;
  Json::Value output_messages;

  static ResponsesResponse fromJson(const Json::Value& json) {
    using namespace json_field;

    uint32_t tid = json.isMember("task_id") && json["task_id"].isUInt()
                       ? json["task_id"].asUInt()
                       : tt::utils::TaskIDGenerator::generate();

    ResponsesResponse r(tid);

    if (json.isMember("id"))
      r.id = getString(json["id"], "id");
    if (json.isMember("created_at") && json["created_at"].isIntegral())
      r.created_at = json["created_at"].asInt64();

    if (json.isMember("incomplete_details") &&
        json["incomplete_details"].isObject()) {
      r.incomplete_details =
          IncompleteDetails::fromJson(json["incomplete_details"]);
    }

    if (json.isMember("instructions") && !json["instructions"].isNull())
      r.instructions = getString(json["instructions"], "instructions");

    if (json.isMember("metadata") && json["metadata"].isObject())
      r.metadata = json["metadata"];

    if (json.isMember("model"))
      r.model = getString(json["model"], "model");
    if (json.isMember("object"))
      r.object = getString(json["object"], "object");

    if (json.isMember("output"))
      r.output = json["output"];

    if (json.isMember("parallel_tool_calls"))
      r.parallel_tool_calls =
          getBool(json["parallel_tool_calls"], "parallel_tool_calls");
    if (json.isMember("temperature"))
      r.temperature = getFloat(json["temperature"], "temperature");

    if (json.isMember("tool_choice"))
      r.tool_choice = json["tool_choice"];
    if (json.isMember("tools"))
      r.tools = json["tools"];

    if (json.isMember("top_p"))
      r.top_p = getFloat(json["top_p"], "top_p");
    if (json.isMember("background"))
      r.background = getBool(json["background"], "background");
    if (json.isMember("max_output_tokens"))
      r.max_output_tokens =
          getInt(json["max_output_tokens"], "max_output_tokens");
    if (json.isMember("max_tool_calls") && !json["max_tool_calls"].isNull())
      r.max_tool_calls = getInt(json["max_tool_calls"], "max_tool_calls");

    if (json.isMember("previous_response_id") &&
        !json["previous_response_id"].isNull())
      r.previous_response_id =
          getString(json["previous_response_id"], "previous_response_id");

    if (json.isMember("prompt") && !json["prompt"].isNull())
      r.prompt = json["prompt"];
    if (json.isMember("reasoning") && !json["reasoning"].isNull())
      r.reasoning = json["reasoning"];

    if (json.isMember("service_tier"))
      r.service_tier = getString(json["service_tier"], "service_tier");
    if (json.isMember("status"))
      r.status = getString(json["status"], "status");

    if (json.isMember("text") && !json["text"].isNull())
      r.text = json["text"];
    if (json.isMember("top_logprobs") && !json["top_logprobs"].isNull())
      r.top_logprobs = getInt(json["top_logprobs"], "top_logprobs");
    if (json.isMember("truncation"))
      r.truncation = getString(json["truncation"], "truncation");

    if (json.isMember("usage") && json["usage"].isObject())
      r.usage = ResponseUsage::fromJson(json["usage"]);

    if (json.isMember("user") && !json["user"].isNull())
      r.user = getString(json["user"], "user");
    if (json.isMember("presence_penalty") && !json["presence_penalty"].isNull())
      r.presence_penalty =
          getFloat(json["presence_penalty"], "presence_penalty");
    if (json.isMember("frequency_penalty") &&
        !json["frequency_penalty"].isNull())
      r.frequency_penalty =
          getFloat(json["frequency_penalty"], "frequency_penalty");

    if (json.isMember("kv_transfer_params") &&
        !json["kv_transfer_params"].isNull())
      r.kv_transfer_params = json["kv_transfer_params"];
    if (json.isMember("input_messages") && !json["input_messages"].isNull())
      r.input_messages = json["input_messages"];
    if (json.isMember("output_messages") && !json["output_messages"].isNull())
      r.output_messages = json["output_messages"];

    return r;
  }

  Json::Value toOpenaiJson() const {
    Json::Value j;
    j["id"] = id;
    j["created_at"] = static_cast<Json::Int64>(created_at);

    if (incomplete_details.has_value())
      j["incomplete_details"] = incomplete_details->toJson();
    if (instructions.has_value())
      j["instructions"] = *instructions;
    if (!metadata.isNull() && !metadata.empty())
      j["metadata"] = metadata;

    j["model"] = model;
    j["object"] = object;
    j["output"] = output;

    j["parallel_tool_calls"] = parallel_tool_calls;
    j["temperature"] = temperature;
    j["tool_choice"] = tool_choice;
    j["tools"] = tools;
    j["top_p"] = top_p;
    j["background"] = background;
    j["max_output_tokens"] = max_output_tokens;
    if (max_tool_calls.has_value())
      j["max_tool_calls"] = *max_tool_calls;
    if (previous_response_id.has_value())
      j["previous_response_id"] = *previous_response_id;
    if (!prompt.isNull())
      j["prompt"] = prompt;
    if (!reasoning.isNull())
      j["reasoning"] = reasoning;

    j["service_tier"] = service_tier;
    j["status"] = status;

    if (!text.isNull())
      j["text"] = text;
    if (top_logprobs.has_value())
      j["top_logprobs"] = *top_logprobs;
    j["truncation"] = truncation;

    if (usage.has_value())
      j["usage"] = usage->toJson();
    if (user.has_value())
      j["user"] = *user;
    if (presence_penalty.has_value())
      j["presence_penalty"] = *presence_penalty;
    if (frequency_penalty.has_value())
      j["frequency_penalty"] = *frequency_penalty;

    if (!kv_transfer_params.isNull())
      j["kv_transfer_params"] = kv_transfer_params;
    if (!input_messages.isNull())
      j["input_messages"] = input_messages;
    if (!output_messages.isNull())
      j["output_messages"] = output_messages;

    return j;
  }

  static ResponsesResponse fromRequest(
      uint32_t taskId, const ResponsesRequest& request,
      const tt::runners::llm_engine::SamplingParams& samplingParams,
      std::string modelName, int64_t createdTime, Json::Value output,
      std::string status, std::optional<ResponseUsage> usage = std::nullopt,
      Json::Value inputMessages = {},
      Json::Value outputMessages = {},
      Json::Value kvTransfer = {}) {
    ResponsesResponse r(taskId);
    r.id = request.request_id.value_or("");
    r.created_at = createdTime;

    if (status == "incomplete") {
      IncompleteDetails d;
      d.reason = "max_output_tokens";
      r.incomplete_details = d;
    }

    r.instructions = request.instructions;
    r.metadata = request.metadata;
    r.model = std::move(modelName);
    r.object = "response";
    r.output = std::move(output);

    r.parallel_tool_calls = request.parallel_tool_calls.value_or(true);
    r.temperature = samplingParams.temperature;
    r.tool_choice = request.tool_choice;
    r.tools = request.tools;
    r.top_p = samplingParams.top_p.value_or(1.0f);
    r.background = request.background.value_or(false);
    r.max_output_tokens = samplingParams.max_tokens.value_or(0);
    if (samplingParams.prompt_logprobs.has_value())
      r.top_logprobs = *samplingParams.prompt_logprobs;
    r.max_tool_calls = request.max_tool_calls;
    r.previous_response_id = request.previous_response_id;
    r.prompt = request.prompt;
    r.reasoning = request.reasoning;
    r.service_tier = request.service_tier.value_or("auto");
    r.status = std::move(status);
    r.text = request.text;

    r.truncation = request.truncation.value_or("disabled");
    r.user = request.user;
    r.usage = std::move(usage);
    r.presence_penalty = samplingParams.presence_penalty;
    r.frequency_penalty = samplingParams.frequency_penalty;

    if (!kvTransfer.isNull())
      r.kv_transfer_params = std::move(kvTransfer);
    if (!inputMessages.isNull())
      r.input_messages = std::move(inputMessages);
    if (!outputMessages.isNull())
      r.output_messages = std::move(outputMessages);

    return r;
  }
};

}  // namespace tt::domain
