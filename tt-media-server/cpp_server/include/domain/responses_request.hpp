// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "domain/base_request.hpp"
#include "domain/chat_message.hpp"
#include "domain/json_field.hpp"
#include "domain/llm_request.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::domain {

/**
 * OpenAI-compatible Responses API request body plus vLLM extensions.
 * Mirrors vllm ResponsesRequest (protocol.py).
 */
struct ResponsesRequest : BaseRequest {
  using BaseRequest::BaseRequest;

  std::optional<bool> background;
  std::optional<std::vector<std::string>> include;
  Json::Value input;

  std::optional<std::string> instructions;
  std::optional<int> max_output_tokens;
  std::optional<int> max_tool_calls;
  Json::Value metadata = Json::Value(Json::objectValue);

  std::optional<std::string> model;
  std::map<std::string, double> logit_bias;
  std::optional<bool> parallel_tool_calls;

  std::optional<std::string> previous_response_id;
  Json::Value prompt;
  Json::Value reasoning;

  std::optional<std::string> service_tier;
  std::optional<bool> store;
  std::optional<bool> stream;

  std::optional<float> temperature;
  Json::Value text;
  Json::Value tool_choice;
  Json::Value tools = Json::Value(Json::arrayValue);

  std::optional<int> top_logprobs;
  std::optional<float> top_p;
  std::optional<int> top_k;
  std::optional<std::string> truncation;

  std::optional<std::string> user;
  std::optional<bool> skip_special_tokens;
  std::optional<bool> include_stop_str_in_output;
  std::optional<float> presence_penalty;
  std::optional<float> frequency_penalty;
  std::optional<std::string> prompt_cache_key;

  std::optional<std::string> request_id;
  Json::Value media_io_kwargs;
  Json::Value mm_processor_kwargs;
  int64_t priority = 0;
  std::optional<std::string> cache_salt;
  std::optional<bool> enable_response_messages;
  Json::Value previous_input_messages;
  Json::Value structured_outputs;

  std::optional<float> repetition_penalty;
  std::optional<int64_t> seed;
  std::vector<std::string> stop;
  std::optional<bool> ignore_eos;
  Json::Value vllm_xargs;
  Json::Value kv_transfer_params;

  static ResponsesRequest fromJson(const Json::Value& json, uint32_t taskId) {
    using namespace json_field;

    ResponsesRequest req(taskId);

    if (json.isMember("background"))
      req.background = getBool(json["background"], "background");

    if (json.isMember("include") && json["include"].isArray()) {
      req.include.emplace();
      for (const auto& item : json["include"]) {
        req.include->push_back(getString(item, "include[]"));
      }
    }

    if (!json.isMember("input"))
      throw std::invalid_argument("input is required");
    const Json::Value& in = json["input"];
    if (!in.isString() && !in.isArray())
      throw std::invalid_argument("input must be a string or array");
    req.input = in;

    if (json.isMember("instructions") && !json["instructions"].isNull())
      req.instructions = getString(json["instructions"], "instructions");
    if (json.isMember("max_output_tokens") &&
        !json["max_output_tokens"].isNull())
      req.max_output_tokens =
          getInt(json["max_output_tokens"], "max_output_tokens");
    if (json.isMember("max_tool_calls") && !json["max_tool_calls"].isNull())
      req.max_tool_calls = getInt(json["max_tool_calls"], "max_tool_calls");

    if (json.isMember("metadata") && json["metadata"].isObject())
      req.metadata = json["metadata"];

    if (json.isMember("model") && !json["model"].isNull())
      req.model = getString(json["model"], "model");

    if (json.isMember("logit_bias") && json["logit_bias"].isObject()) {
      const auto& names = json["logit_bias"].getMemberNames();
      for (const auto& name : names) {
        const Json::Value& v = json["logit_bias"][name];
        if (!v.isNumeric())
          throw std::invalid_argument("logit_bias values must be numbers");
        req.logit_bias[name] = v.asDouble();
      }
    }

    if (json.isMember("parallel_tool_calls") &&
        !json["parallel_tool_calls"].isNull())
      req.parallel_tool_calls =
          getBool(json["parallel_tool_calls"], "parallel_tool_calls");

    if (json.isMember("previous_response_id") &&
        !json["previous_response_id"].isNull())
      req.previous_response_id =
          getString(json["previous_response_id"], "previous_response_id");

    if (json.isMember("prompt") && !json["prompt"].isNull())
      req.prompt = json["prompt"];
    if (json.isMember("reasoning") && !json["reasoning"].isNull())
      req.reasoning = json["reasoning"];

    if (json.isMember("service_tier") && !json["service_tier"].isNull())
      req.service_tier = getString(json["service_tier"], "service_tier");
    if (json.isMember("store") && !json["store"].isNull())
      req.store = getBool(json["store"], "store");
    if (json.isMember("stream") && !json["stream"].isNull())
      req.stream = getBool(json["stream"], "stream");

    if (json.isMember("temperature") && !json["temperature"].isNull())
      req.temperature = getFloat(json["temperature"], "temperature");
    if (json.isMember("text") && !json["text"].isNull())
      req.text = json["text"];
    if (json.isMember("tool_choice") && !json["tool_choice"].isNull())
      req.tool_choice = json["tool_choice"];
    if (json.isMember("tools"))
      req.tools = json["tools"];

    if (json.isMember("top_logprobs") && !json["top_logprobs"].isNull())
      req.top_logprobs = getInt(json["top_logprobs"], "top_logprobs");
    if (json.isMember("top_p") && !json["top_p"].isNull())
      req.top_p = getFloat(json["top_p"], "top_p");
    if (json.isMember("top_k") && !json["top_k"].isNull())
      req.top_k = getInt(json["top_k"], "top_k");
    if (json.isMember("truncation") && !json["truncation"].isNull())
      req.truncation = getString(json["truncation"], "truncation");

    if (json.isMember("user") && !json["user"].isNull())
      req.user = getString(json["user"], "user");
    if (json.isMember("skip_special_tokens") &&
        !json["skip_special_tokens"].isNull())
      req.skip_special_tokens =
          getBool(json["skip_special_tokens"], "skip_special_tokens");
    if (json.isMember("include_stop_str_in_output") &&
        !json["include_stop_str_in_output"].isNull())
      req.include_stop_str_in_output =
          getBool(json["include_stop_str_in_output"],
                  "include_stop_str_in_output");
    if (json.isMember("presence_penalty") && !json["presence_penalty"].isNull())
      req.presence_penalty =
          getFloat(json["presence_penalty"], "presence_penalty");
    if (json.isMember("frequency_penalty") &&
        !json["frequency_penalty"].isNull())
      req.frequency_penalty =
          getFloat(json["frequency_penalty"], "frequency_penalty");
    if (json.isMember("prompt_cache_key") && !json["prompt_cache_key"].isNull())
      req.prompt_cache_key =
          getString(json["prompt_cache_key"], "prompt_cache_key");

    if (json.isMember("request_id") && !json["request_id"].isNull())
      req.request_id = getString(json["request_id"], "request_id");
    else
      req.request_id = makeRequestId();

    if (json.isMember("media_io_kwargs") && !json["media_io_kwargs"].isNull())
      req.media_io_kwargs = json["media_io_kwargs"];
    if (json.isMember("mm_processor_kwargs") &&
        !json["mm_processor_kwargs"].isNull())
      req.mm_processor_kwargs = json["mm_processor_kwargs"];

    if (json.isMember("priority")) {
      const Json::Value& p = json["priority"];
      if (!p.isIntegral())
        throw std::invalid_argument("priority must be an integer");
      req.priority = p.asInt64();
    }

    if (json.isMember("cache_salt") && !json["cache_salt"].isNull())
      req.cache_salt = getString(json["cache_salt"], "cache_salt");
    if (json.isMember("enable_response_messages") &&
        !json["enable_response_messages"].isNull())
      req.enable_response_messages =
          getBool(json["enable_response_messages"], "enable_response_messages");
    if (json.isMember("previous_input_messages") &&
        !json["previous_input_messages"].isNull())
      req.previous_input_messages = json["previous_input_messages"];
    if (json.isMember("structured_outputs") &&
        !json["structured_outputs"].isNull())
      req.structured_outputs = json["structured_outputs"];

    if (json.isMember("repetition_penalty") &&
        !json["repetition_penalty"].isNull())
      req.repetition_penalty =
          getFloat(json["repetition_penalty"], "repetition_penalty");
    if (json.isMember("seed") && !json["seed"].isNull()) {
      const Json::Value& s = json["seed"];
      if (!s.isIntegral())
        throw std::invalid_argument("seed must be an integer");
      req.seed = s.asInt64();
    }

    if (json.isMember("stop")) {
      const Json::Value& sv = json["stop"];
      if (sv.isString()) {
        req.stop.push_back(sv.asString());
      } else if (sv.isArray()) {
        for (const auto& s : sv)
          req.stop.push_back(getString(s, "stop[]"));
      } else if (!sv.isNull()) {
        throw std::invalid_argument("stop must be a string, array, or null");
      }
    }

    if (json.isMember("ignore_eos") && !json["ignore_eos"].isNull())
      req.ignore_eos = getBool(json["ignore_eos"], "ignore_eos");
    if (json.isMember("vllm_xargs") && !json["vllm_xargs"].isNull())
      req.vllm_xargs = json["vllm_xargs"];
    if (json.isMember("kv_transfer_params") &&
        !json["kv_transfer_params"].isNull())
      req.kv_transfer_params = json["kv_transfer_params"];

    return req;
  }

  Json::Value toJson() const {
    Json::Value j;
    j["task_id"] = task_id;

    if (background.has_value()) j["background"] = *background;
    if (include.has_value()) {
      Json::Value arr(Json::arrayValue);
      for (const auto& s : *include) arr.append(s);
      j["include"] = std::move(arr);
    }
    j["input"] = input;

    if (instructions.has_value()) j["instructions"] = *instructions;
    if (max_output_tokens.has_value())
      j["max_output_tokens"] = *max_output_tokens;
    if (max_tool_calls.has_value()) j["max_tool_calls"] = *max_tool_calls;
    if (!metadata.isNull() && !metadata.empty()) j["metadata"] = metadata;

    if (model.has_value()) j["model"] = *model;
    if (!logit_bias.empty()) {
      Json::Value lb(Json::objectValue);
      for (const auto& [k, v] : logit_bias) lb[k] = v;
      j["logit_bias"] = std::move(lb);
    }

    if (parallel_tool_calls.has_value())
      j["parallel_tool_calls"] = *parallel_tool_calls;
    if (previous_response_id.has_value())
      j["previous_response_id"] = *previous_response_id;
    if (!prompt.isNull()) j["prompt"] = prompt;
    if (!reasoning.isNull()) j["reasoning"] = reasoning;

    if (service_tier.has_value()) j["service_tier"] = *service_tier;
    if (store.has_value()) j["store"] = *store;
    if (stream.has_value()) j["stream"] = *stream;

    if (temperature.has_value()) j["temperature"] = *temperature;
    if (!text.isNull()) j["text"] = text;
    if (!tool_choice.isNull()) j["tool_choice"] = tool_choice;
    j["tools"] = tools;

    if (top_logprobs.has_value()) j["top_logprobs"] = *top_logprobs;
    if (top_p.has_value()) j["top_p"] = *top_p;
    if (top_k.has_value()) j["top_k"] = *top_k;
    if (truncation.has_value()) j["truncation"] = *truncation;

    if (user.has_value()) j["user"] = *user;
    if (skip_special_tokens.has_value())
      j["skip_special_tokens"] = *skip_special_tokens;
    if (include_stop_str_in_output.has_value())
      j["include_stop_str_in_output"] = *include_stop_str_in_output;
    if (presence_penalty.has_value())
      j["presence_penalty"] = *presence_penalty;
    if (frequency_penalty.has_value())
      j["frequency_penalty"] = *frequency_penalty;
    if (prompt_cache_key.has_value())
      j["prompt_cache_key"] = *prompt_cache_key;

    if (request_id.has_value()) j["request_id"] = *request_id;
    if (!media_io_kwargs.isNull()) j["media_io_kwargs"] = media_io_kwargs;
    if (!mm_processor_kwargs.isNull())
      j["mm_processor_kwargs"] = mm_processor_kwargs;
    j["priority"] = static_cast<Json::Int64>(priority);
    if (cache_salt.has_value()) j["cache_salt"] = *cache_salt;
    if (enable_response_messages.has_value())
      j["enable_response_messages"] = *enable_response_messages;
    if (!previous_input_messages.isNull())
      j["previous_input_messages"] = previous_input_messages;
    if (!structured_outputs.isNull())
      j["structured_outputs"] = structured_outputs;

    if (repetition_penalty.has_value())
      j["repetition_penalty"] = *repetition_penalty;
    if (seed.has_value()) j["seed"] = static_cast<Json::Int64>(*seed);
    if (!stop.empty()) {
      Json::Value arr(Json::arrayValue);
      for (const auto& s : stop) arr.append(s);
      j["stop"] = std::move(arr);
    }
    if (ignore_eos.has_value()) j["ignore_eos"] = *ignore_eos;
    if (!vllm_xargs.isNull()) j["vllm_xargs"] = vllm_xargs;
    if (!kv_transfer_params.isNull())
      j["kv_transfer_params"] = kv_transfer_params;

    return j;
  }

  /**
   * Convert the Responses API input into ChatMessages.
   *
   * If input is a plain string, it becomes a single user message.
   * If input is an array, each element with a "role" and text "content"
   * is converted; elements that are plain strings become user messages.
   */
  std::vector<ChatMessage> toMessages() const {
    std::vector<ChatMessage> msgs;

    if (instructions.has_value()) {
      ChatMessage sys;
      sys.role = "system";
      sys.content = *instructions;
      msgs.push_back(std::move(sys));
    }

    if (input.isString()) {
      ChatMessage m;
      m.role = "user";
      m.content = input.asString();
      msgs.push_back(std::move(m));
      return msgs;
    }

    if (input.isArray()) {
      for (const auto& item : input) {
        if (item.isString()) {
          ChatMessage m;
          m.role = "user";
          m.content = item.asString();
          msgs.push_back(std::move(m));
        } else if (item.isObject()) {
          msgs.push_back(ChatMessage::fromJson(item));
        }
      }
    }
    return msgs;
  }

  /** Convert to LLMRequest: applies chat template to the input, then copies
   * sampling parameters for the LLM pipeline. */
  LLMRequest toLLMRequest() const {
    LLMRequest out(task_id);
    out.model = model;
    out.prompt = tt::utils::tokenizers::activeTokenizer().applyChatTemplate(
        toMessages());

    out.max_tokens = max_output_tokens;
    out.presence_penalty = presence_penalty.value_or(0.0f);
    out.frequency_penalty = frequency_penalty.value_or(0.0f);
    out.stream = stream.value_or(false);
    out.stop = stop;
    out.temperature = temperature;
    out.top_p = top_p;
    out.top_k = top_k;
    out.user = user;
    out.repetition_penalty = repetition_penalty;
    out.include_stop_str_in_output =
        include_stop_str_in_output.value_or(false);
    out.ignore_eos = ignore_eos.value_or(false);
    out.skip_special_tokens = skip_special_tokens.value_or(true);

    if (seed.has_value()) {
      int64_t s = *seed;
      if (s >= std::numeric_limits<int>::min() &&
          s <= std::numeric_limits<int>::max()) {
        out.seed = static_cast<int>(s);
      }
    }

    if (top_logprobs.has_value()) {
      out.logprobs = *top_logprobs;
    }

    if (truncation.has_value() && *truncation != "disabled") {
      out.truncate_prompt_tokens = -1;
    }

    return out;
  }

 private:
  static std::string makeRequestId() {
    static std::mutex mtx;
    static std::random_device rd;
    static std::mt19937_64 gen(rd());

    std::lock_guard<std::mutex> lock(mtx);
    uint64_t part1 = gen();
    uint64_t part2 = gen();

    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    ss << std::setw(8) << (part1 & 0xFFFFFFFF) << '-';
    ss << std::setw(4) << ((part1 >> 32) & 0xFFFF) << '-';
    ss << "4" << std::setw(3) << ((part1 >> 48) & 0x0FFF) << '-';
    ss << std::setw(1) << (8 | ((part2 & 0x3))) << std::setw(3)
       << ((part2 >> 2) & 0xFFF) << '-';
    ss << std::setw(12) << ((part2 >> 14) & 0xFFFFFFFFFFFF);
    return "resp_" + ss.str();
  }
};

}  // namespace tt::domain
