// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "domain/base_request.hpp"
#include "domain/chat_message.hpp"
#include "domain/json_field.hpp"
#include "domain/llm_request.hpp"
#include "domain/response_format.hpp"
#include "domain/tool_calls/tool.hpp"
#include "domain/tool_calls/tool_choice.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::domain {

/** Legacy format: "Role: content\n\n" per message, ending with "Assistant: ".
 */
inline std::string messagesToPrompt(const std::vector<ChatMessage>& messages) {
  std::ostringstream out;
  for (const auto& m : messages) {
    std::string role = m.role.empty() ? "user" : m.role;
    out << (role == "system"      ? "System"
            : role == "user"      ? "User"
            : role == "assistant" ? "Assistant"
                                  : role)
        << ": " << m.content << "\n\n";
  }
  out << "Assistant: ";
  return out.str();
}

struct ChatCompletionRequest : BaseRequest {
  using BaseRequest::BaseRequest;
  std::optional<std::string> model;

  std::vector<ChatMessage> messages;

  bool echo = false;
  std::optional<int> max_tokens;
  int n = 1;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  std::optional<std::string> suffix;
  bool stream = false;
  std::optional<StreamOptions> stream_options;

  std::vector<std::string> stop;
  std::optional<int> seed;
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<int> logprobs;
  std::optional<std::string> user;

  bool use_beam_search = false;
  std::optional<int> top_k;
  std::optional<float> min_p;
  std::optional<float> repetition_penalty;
  float length_penalty = 1.0f;
  std::vector<int> stop_token_ids;
  bool include_stop_str_in_output = false;
  bool ignore_eos = false;
  int min_tokens = 0;
  bool skip_special_tokens = true;
  bool spaces_between_special_tokens = true;
  std::optional<std::vector<int>> allowed_token_ids;
  std::optional<int> prompt_logprobs;
  std::optional<int> truncate_prompt_tokens;

  bool fast_mode = false;

  // Structured output constraint
  std::optional<ResponseFormat> response_format;

  // Session management
  std::optional<std::string> sessionId;

  // Tool calling support
  std::optional<std::vector<tool_calls::Tool>> tools;
  std::optional<tool_calls::ToolChoice> tool_choice;
  bool parallel_tool_calls = true;

  static ChatCompletionRequest fromJson(const Json::Value& json,
                                        uint32_t taskId) {
    ChatCompletionRequest req(std::move(taskId));
    using namespace json_field;

    if (json.isMember("model") && !json["model"].isNull()) {
      req.model = getString(json["model"], "model");
    }

    if (json.isMember("messages")) {
      checkArray(json["messages"], "messages");
      for (const auto& m : json["messages"]) {
        req.messages.push_back(ChatMessage::fromJson(m));
      }
    }

    if (json.isMember("echo")) req.echo = getBool(json["echo"], "echo");
    if (json.isMember("max_tokens"))
      req.max_tokens = getInt(json["max_tokens"], "max_tokens");
    if (json.isMember("max_completion_tokens"))
      req.max_tokens =
          getInt(json["max_completion_tokens"], "max_completion_tokens");
    if (json.isMember("n")) req.n = getInt(json["n"], "n");
    if (json.isMember("presence_penalty"))
      req.presence_penalty =
          getFloat(json["presence_penalty"], "presence_penalty");
    if (json.isMember("frequency_penalty"))
      req.frequency_penalty =
          getFloat(json["frequency_penalty"], "frequency_penalty");
    if (json.isMember("suffix") && !json["suffix"].isNull()) {
      req.suffix = getString(json["suffix"], "suffix");
    }
    if (json.isMember("stream")) req.stream = getBool(json["stream"], "stream");
    if (json.isMember("stream_options") && !json["stream_options"].isNull()) {
      checkObject(json["stream_options"], "stream_options");
      req.stream_options = StreamOptions::fromJson(json["stream_options"]);
    }

    if (json.isMember("stop")) {
      const auto& stopVal = json["stop"];
      if (stopVal.isString()) {
        req.stop.push_back(stopVal.asString());
      } else if (stopVal.isArray()) {
        for (const auto& s : stopVal)
          req.stop.push_back(getString(s, "stop[]"));
      } else {
        throw std::invalid_argument(
            "stop must be a string or array of strings");
      }
    }

    if (json.isMember("seed") && !json["seed"].isNull())
      req.seed = getInt(json["seed"], "seed");
    if (json.isMember("temperature") && !json["temperature"].isNull())
      req.temperature = getFloat(json["temperature"], "temperature");
    if (json.isMember("top_p") && !json["top_p"].isNull())
      req.top_p = getFloat(json["top_p"], "top_p");
    if (json.isMember("logprobs") && !json["logprobs"].isNull())
      req.logprobs = getInt(json["logprobs"], "logprobs");
    if (json.isMember("user") && !json["user"].isNull())
      req.user = getString(json["user"], "user");

    if (json.isMember("use_beam_search"))
      req.use_beam_search = getBool(json["use_beam_search"], "use_beam_search");
    if (json.isMember("top_k") && !json["top_k"].isNull())
      req.top_k = getInt(json["top_k"], "top_k");
    if (json.isMember("min_p") && !json["min_p"].isNull())
      req.min_p = getFloat(json["min_p"], "min_p");
    if (json.isMember("repetition_penalty") &&
        !json["repetition_penalty"].isNull())
      req.repetition_penalty =
          getFloat(json["repetition_penalty"], "repetition_penalty");
    if (json.isMember("length_penalty"))
      req.length_penalty = getFloat(json["length_penalty"], "length_penalty");

    if (json.isMember("stop_token_ids") && json["stop_token_ids"].isArray()) {
      for (const auto& id : json["stop_token_ids"])
        req.stop_token_ids.push_back(getInt(id, "stop_token_ids[]"));
    }

    if (json.isMember("include_stop_str_in_output"))
      req.include_stop_str_in_output = getBool(
          json["include_stop_str_in_output"], "include_stop_str_in_output");
    if (json.isMember("ignore_eos"))
      req.ignore_eos = getBool(json["ignore_eos"], "ignore_eos");
    if (json.isMember("min_tokens"))
      req.min_tokens = getInt(json["min_tokens"], "min_tokens");
    if (json.isMember("skip_special_tokens"))
      req.skip_special_tokens =
          getBool(json["skip_special_tokens"], "skip_special_tokens");
    if (json.isMember("spaces_between_special_tokens"))
      req.spaces_between_special_tokens =
          getBool(json["spaces_between_special_tokens"],
                  "spaces_between_special_tokens");

    if (json.isMember("fast_mode")) req.fast_mode = json["fast_mode"].asBool();

    if (json.isMember("response_format") && !json["response_format"].isNull()) {
      req.response_format = ResponseFormat::fromJson(json["response_format"]);
    }

    if (json.isMember("session_id") && !json["session_id"].isNull())
      req.sessionId = getString(json["session_id"], "session_id");

    if (json.isMember("tool_choice") && !json["tool_choice"].isNull()) {
      req.tool_choice = tool_calls::ToolChoice::fromJson(json["tool_choice"]);
    }

    if (json.isMember("tools") && json["tools"].isArray()) {
      std::vector<tool_calls::Tool> toolList;
      for (const auto& tool : json["tools"]) {
        toolList.push_back(tool_calls::Tool::fromJson(tool));
      }
      req.tools = toolList;
    }
    if (json.isMember("parallel_tool_calls") &&
        !json["parallel_tool_calls"].isNull())
      req.parallel_tool_calls =
          getBool(json["parallel_tool_calls"], "parallel_tool_calls");

    validateToolFields(req);
    return req;
  }

  std::string toString() const {
    std::string lastMsg;
    if (!messages.empty()) {
      const auto& m = messages.back();
      lastMsg = m.role + ": \"" +
                detail::truncate(m.content, detail::MAX_PROMPT_LOG_LENGTH) +
                "\"";
    }

    std::ostringstream out;
    out << "task_id=" << task_id << " model=" << model.value_or("default")
        << " stream=" << stream << " messages=" << messages.size()
        << " last_msg=[" << lastMsg << "]"
        << " max_tokens=" << detail::optStr(max_tokens)
        << " temperature=" << detail::optStr(temperature)
        << " top_p=" << detail::optStr(top_p)
        << " top_k=" << detail::optStr(top_k)
        << " min_p=" << detail::optStr(min_p)
        << " presence_penalty=" << presence_penalty
        << " frequency_penalty=" << frequency_penalty << " n=" << n
        << " stop_count=" << stop.size();
    return out.str();
  }

  /** Convert to LLMRequest: applies chat template to messages, then copies
   * sampling parameters for the LLM pipeline. */
  LLMRequest toLLMRequest() const {
    LLMRequest out(task_id);
    out.model = model;
    out.messages = messages;
    out.prompt = tt::utils::tokenizers::activeTokenizer().applyChatTemplate(
        messages, true, tools);

    out.echo = echo;
    out.max_tokens = max_tokens;
    out.n = n;
    out.presence_penalty = presence_penalty;
    out.frequency_penalty = frequency_penalty;
    out.suffix = suffix;
    out.stream = stream;
    out.stream_options = stream_options;
    out.stop = stop;
    out.seed = seed;
    out.temperature = temperature;
    out.top_p = top_p;
    out.logprobs = logprobs;
    out.user = user;
    out.use_beam_search = use_beam_search;
    out.top_k = top_k;
    out.min_p = min_p;
    out.repetition_penalty = repetition_penalty;
    out.length_penalty = length_penalty;
    out.stop_token_ids = stop_token_ids;
    out.parallel_tool_calls = parallel_tool_calls;
    out.tools = tools;
    out.tool_choice = tool_choice;
    out.include_stop_str_in_output = include_stop_str_in_output;
    out.ignore_eos = ignore_eos;
    out.min_tokens = min_tokens;
    out.skip_special_tokens = skip_special_tokens;
    out.spaces_between_special_tokens = spaces_between_special_tokens;
    out.allowed_token_ids = allowed_token_ids;
    out.prompt_logprobs = prompt_logprobs;
    out.truncate_prompt_tokens = truncate_prompt_tokens;
    out.fast_mode = fast_mode;
    out.response_format = response_format;
    out.sessionId = sessionId;
    return out;
  }

 private:
  static void validateToolFields(const ChatCompletionRequest& req) {
    if (!req.tool_choice.has_value()) return;

    const auto& toolChoice = req.tool_choice.value();
    const auto& type = toolChoice.type;
    const bool toolsMissing = !req.tools.has_value() || req.tools->empty();

    if (toolsMissing && type != "none") {
      throw std::invalid_argument("tool_choice='" + type +
                                  "' requires non-empty 'tools'");
    }
  }
};

}  // namespace tt::domain
