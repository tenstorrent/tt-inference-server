// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "domain/base_request.hpp"
#include "domain/chat_message.hpp"
#include "domain/llm_request.hpp"
#include "utils/tokenizer.hpp"

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

  // Session management
  std::optional<std::string> sessionId;

  static ChatCompletionRequest fromJson(const Json::Value& json,
                                        uint32_t taskId) {
    ChatCompletionRequest req(std::move(taskId));

    if (json.isMember("model") && !json["model"].isNull()) {
      req.model = json["model"].asString();
    }

    if (json.isMember("messages") && json["messages"].isArray()) {
      for (const auto& m : json["messages"]) {
        req.messages.push_back(ChatMessage::fromJson(m));
      }
    }

    if (json.isMember("echo")) req.echo = json["echo"].asBool();
    if (json.isMember("max_tokens"))
      req.max_tokens = json["max_tokens"].asInt();
    if (json.isMember("max_completion_tokens"))
      req.max_tokens = json["max_completion_tokens"].asInt();
    if (json.isMember("n")) req.n = json["n"].asInt();
    if (json.isMember("presence_penalty"))
      req.presence_penalty = json["presence_penalty"].asFloat();
    if (json.isMember("frequency_penalty"))
      req.frequency_penalty = json["frequency_penalty"].asFloat();
    if (json.isMember("suffix") && !json["suffix"].isNull()) {
      req.suffix = json["suffix"].asString();
    }
    if (json.isMember("stream")) req.stream = json["stream"].asBool();
    if (json.isMember("stream_options") && !json["stream_options"].isNull()) {
      req.stream_options = StreamOptions::fromJson(json["stream_options"]);
    }

    if (json.isMember("stop")) {
      if (json["stop"].isString()) {
        req.stop.push_back(json["stop"].asString());
      } else if (json["stop"].isArray()) {
        for (const auto& s : json["stop"]) {
          req.stop.push_back(s.asString());
        }
      }
    }

    if (json.isMember("seed") && !json["seed"].isNull()) {
      req.seed = json["seed"].asInt();
    }
    if (json.isMember("temperature") && !json["temperature"].isNull()) {
      req.temperature = json["temperature"].asFloat();
    }
    if (json.isMember("top_p") && !json["top_p"].isNull()) {
      req.top_p = json["top_p"].asFloat();
    }
    if (json.isMember("logprobs") && !json["logprobs"].isNull()) {
      req.logprobs = json["logprobs"].asInt();
    }
    if (json.isMember("user") && !json["user"].isNull()) {
      req.user = json["user"].asString();
    }

    if (json.isMember("use_beam_search"))
      req.use_beam_search = json["use_beam_search"].asBool();
    if (json.isMember("top_k") && !json["top_k"].isNull()) {
      req.top_k = json["top_k"].asInt();
    }
    if (json.isMember("min_p") && !json["min_p"].isNull()) {
      req.min_p = json["min_p"].asFloat();
    }
    if (json.isMember("repetition_penalty") &&
        !json["repetition_penalty"].isNull()) {
      req.repetition_penalty = json["repetition_penalty"].asFloat();
    }
    if (json.isMember("length_penalty"))
      req.length_penalty = json["length_penalty"].asFloat();

    if (json.isMember("stop_token_ids") && json["stop_token_ids"].isArray()) {
      for (const auto& id : json["stop_token_ids"]) {
        req.stop_token_ids.push_back(id.asInt());
      }
    }

    if (json.isMember("include_stop_str_in_output")) {
      req.include_stop_str_in_output =
          json["include_stop_str_in_output"].asBool();
    }
    if (json.isMember("ignore_eos"))
      req.ignore_eos = json["ignore_eos"].asBool();
    if (json.isMember("min_tokens"))
      req.min_tokens = json["min_tokens"].asInt();
    if (json.isMember("skip_special_tokens"))
      req.skip_special_tokens = json["skip_special_tokens"].asBool();
    if (json.isMember("spaces_between_special_tokens")) {
      req.spaces_between_special_tokens =
          json["spaces_between_special_tokens"].asBool();
    }

    if (json.isMember("session_id") && !json["session_id"].isNull()) {
      req.sessionId = json["session_id"].asString();
    }

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
    out.prompt = tt::utils::activeTokenizer().applyChatTemplate(messages);

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
    out.include_stop_str_in_output = include_stop_str_in_output;
    out.ignore_eos = ignore_eos;
    out.min_tokens = min_tokens;
    out.skip_special_tokens = skip_special_tokens;
    out.spaces_between_special_tokens = spaces_between_special_tokens;
    out.allowed_token_ids = allowed_token_ids;
    out.prompt_logprobs = prompt_logprobs;
    out.truncate_prompt_tokens = truncate_prompt_tokens;
    out.sessionId = sessionId;
    return out;
  }
};

}  // namespace tt::domain
