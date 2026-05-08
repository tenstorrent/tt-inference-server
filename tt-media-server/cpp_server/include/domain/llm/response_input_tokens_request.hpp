// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include "domain/base_request.hpp"
#include "domain/json_field.hpp"
#include "domain/responses_request.hpp"

namespace tt::domain::llm {

/**
 * Request body for POST /v1/responses/input_tokens.
 *
 * Mirrors OpenAI "Get input token counts"
 * (https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens/methods/count).
 *
 * Per the OpenAI spec, every field is optional. The server is stateless and
 * cannot resolve `conversation` or `previous_response_id` to prior items, so
 * those fields are accepted and validated but do not contribute to the token
 * count. Token-affecting fields (`input`, `instructions`, `model`, `tools`,
 * `tool_choice`, `text`, `reasoning`, `truncation`, `parallel_tool_calls`)
 * are forwarded into a `ResponsesRequest` so the count reflects what
 * `POST /v1/responses` would send to the model.
 */
struct ResponseInputTokensRequest : BaseRequest {
  using BaseRequest::BaseRequest;

  // string or { id: string }
  Json::Value conversation;
  std::optional<std::string> instructions;
  // string or array of input items
  Json::Value input;
  std::optional<std::string> model;
  std::optional<bool> parallel_tool_calls;
  std::optional<std::string> previous_response_id;
  Json::Value reasoning;
  Json::Value text;
  Json::Value tool_choice;
  Json::Value tools = Json::Value(Json::arrayValue);
  std::optional<std::string> truncation;

  static ResponseInputTokensRequest fromJson(const Json::Value& json,
                                             uint32_t taskId) {
    using namespace tt::domain::json_field;

    ResponseInputTokensRequest req(taskId);

    if (json.isMember("conversation") && !json["conversation"].isNull()) {
      const Json::Value& c = json["conversation"];
      if (c.isString()) {
        req.conversation = c;
      } else if (c.isObject()) {
        if (!c.isMember("id") || !c["id"].isString())
          throw std::invalid_argument(
              "conversation object must have a string 'id' field");
        req.conversation = c;
      } else {
        throw std::invalid_argument(
            "conversation must be a string or { id } object");
      }
    }

    if (json.isMember("input") && !json["input"].isNull()) {
      const Json::Value& in = json["input"];
      if (!in.isString() && !in.isArray())
        throw std::invalid_argument("input must be a string or array");
      req.input = in;
    }

    if (json.isMember("instructions") && !json["instructions"].isNull())
      req.instructions = getString(json["instructions"], "instructions");

    if (json.isMember("model") && !json["model"].isNull())
      req.model = getString(json["model"], "model");

    if (json.isMember("parallel_tool_calls") &&
        !json["parallel_tool_calls"].isNull())
      req.parallel_tool_calls =
          getBool(json["parallel_tool_calls"], "parallel_tool_calls");

    if (json.isMember("previous_response_id") &&
        !json["previous_response_id"].isNull()) {
      req.previous_response_id =
          getString(json["previous_response_id"], "previous_response_id");
      if (!req.conversation.isNull())
        throw std::invalid_argument(
            "previous_response_id cannot be used together with conversation");
    }

    if (json.isMember("reasoning") && !json["reasoning"].isNull()) {
      checkObject(json["reasoning"], "reasoning");
      req.reasoning = json["reasoning"];
    }

    if (json.isMember("text") && !json["text"].isNull()) {
      checkObject(json["text"], "text");
      req.text = json["text"];
    }

    if (json.isMember("tool_choice") && !json["tool_choice"].isNull())
      req.tool_choice = json["tool_choice"];

    if (json.isMember("tools") && !json["tools"].isNull()) {
      checkArray(json["tools"], "tools");
      req.tools = json["tools"];
    }

    if (json.isMember("truncation") && !json["truncation"].isNull()) {
      const std::string t = getString(json["truncation"], "truncation");
      if (t != "auto" && t != "disabled")
        throw std::invalid_argument("truncation must be 'auto' or 'disabled'");
      req.truncation = t;
    }

    return req;
  }

  std::string toString() const {
    std::string inputType = "none";
    std::size_t inputSize = 0;
    std::string inputPreview;
    if (input.isString()) {
      inputType = "string";
      const std::string s = input.asString();
      inputSize = s.size();
      inputPreview =
          "\"" + detail::truncate(s, detail::MAX_PROMPT_LOG_LENGTH) + "\"";
    } else if (input.isArray()) {
      inputType = "array";
      inputSize = input.size();
    }

    std::string conv = "none";
    if (conversation.isString()) {
      conv = conversation.asString();
    } else if (conversation.isObject() && conversation.isMember("id") &&
               conversation["id"].isString()) {
      conv = conversation["id"].asString();
    }

    std::ostringstream out;
    out << "task_id=" << task_id << " input_type=" << inputType
        << " input_size=" << inputSize << " first_input=[" << inputPreview
        << "]"
        << " conversation=" << conv << " model=" << model.value_or("none")
        << " previous_response_id=" << previous_response_id.value_or("none")
        << " has_instructions=" << (instructions.has_value() ? "true" : "false")
        << " tools_count=" << (tools.isArray() ? tools.size() : 0u)
        << " truncation=" << truncation.value_or("none");
    return out.str();
  }

  Json::Value toJson() const {
    Json::Value j;
    j["task_id"] = task_id;
    if (!conversation.isNull()) j["conversation"] = conversation;
    if (instructions.has_value()) j["instructions"] = *instructions;
    if (!input.isNull()) j["input"] = input;
    if (model.has_value()) j["model"] = *model;
    if (parallel_tool_calls.has_value())
      j["parallel_tool_calls"] = *parallel_tool_calls;
    if (previous_response_id.has_value())
      j["previous_response_id"] = *previous_response_id;
    if (!reasoning.isNull()) j["reasoning"] = reasoning;
    if (!text.isNull()) j["text"] = text;
    if (!tool_choice.isNull()) j["tool_choice"] = tool_choice;
    if (tools.isArray() && !tools.empty()) j["tools"] = tools;
    if (truncation.has_value()) j["truncation"] = *truncation;
    return j;
  }

  /**
   * Build a ResponsesRequest from the input_tokens request body, so the
   * existing tokenization pipeline (chat-template + LLMRequest) can reuse
   * the same conversion logic as POST /v1/responses.
   *
   * Forwards every field that influences the prompt sent to the tokenizer.
   * The stateful fields `conversation` and `previous_response_id` are not
   * forwarded since the server cannot resolve them to prior items.
   */
  ResponsesRequest toResponsesRequest() const {
    ResponsesRequest r(task_id);
    r.input = input;
    r.instructions = instructions;
    r.model = model;
    r.parallel_tool_calls = parallel_tool_calls;
    r.previous_response_id = previous_response_id;
    r.reasoning = reasoning;
    r.text = text;
    r.tool_choice = tool_choice;
    r.tools = tools;
    r.truncation = truncation;
    return r;
  }
};

}  // namespace tt::domain::llm
