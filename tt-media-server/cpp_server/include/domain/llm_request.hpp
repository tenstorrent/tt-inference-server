// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "domain/base_request.hpp"
#include "domain/json_field.hpp"
#include "domain/response_format.hpp"

namespace tt::domain {

namespace detail {

constexpr size_t MAX_PROMPT_LOG_LENGTH = 200;

inline std::string truncate(const std::string& s, size_t maxLen) {
  if (s.size() <= maxLen) return s;
  return s.substr(0, maxLen) + "...(" + std::to_string(s.size()) + " chars)";
}

template <typename T>
std::string optStr(const std::optional<T>& opt) {
  if (!opt.has_value()) return "none";
  if constexpr (std::is_same_v<T, std::string>) {
    return opt.value();
  } else {
    return std::to_string(opt.value());
  }
}

}  // namespace detail

/**
 * Stream options for OpenAI-compatible streaming responses.
 */
struct StreamOptions {
  bool include_usage = true;
  bool continuous_usage_stats = false;

  static StreamOptions fromJson(const Json::Value& json) {
    StreamOptions opts;
    if (json.isMember("include_usage"))
      opts.include_usage =
          json_field::getBool(json["include_usage"], "include_usage");
    if (json.isMember("continuous_usage_stats"))
      opts.continuous_usage_stats = json_field::getBool(
          json["continuous_usage_stats"], "continuous_usage_stats");
    return opts;
  }
};

/**
 * Internal LLM pipeline request.
 * Holds the tokenized prompt and sampling parameters used by LLMService.
 * Created from ChatCompletionRequest via toLLMRequest(), or directly by
 * DisaggregationService from IPC messages.
 */
struct LLMRequest : BaseRequest {
  using BaseRequest::BaseRequest;

  // Model identifier
  std::optional<std::string> model;

  // Prompt can be a string or a list of token ids
  std::variant<std::string, std::vector<int>> prompt;

  // Response configuration
  bool echo = false;
  std::optional<int> max_tokens;
  int n = 1;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  std::optional<std::string> suffix;
  bool stream = false;
  std::optional<StreamOptions> stream_options;

  // Stopping criteria
  std::vector<std::string> stop;

  // Reproducibility
  std::optional<int> seed;

  // Sampling params
  std::optional<float> temperature;
  std::optional<float> top_p;

  // Logging and debugging
  std::optional<int> logprobs;
  std::optional<std::string> user;

  // Completion sampling params
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
  int prompt_tokens_count = 0;
  bool fast_mode = false;

  // Structured output constraint
  std::optional<ResponseFormat> response_format;

  // Session management (internal use only, not parsed from JSON)
  std::optional<std::string> sessionId;
  std::optional<uint32_t> slotId;
  bool continuation =
      false;  // True if this request continues an existing session

  std::string toString() const {
    std::string promptInfo;
    if (auto* s = std::get_if<std::string>(&prompt)) {
      promptInfo =
          "\"" + detail::truncate(*s, detail::MAX_PROMPT_LOG_LENGTH) + "\"";
    } else {
      auto& tokens = std::get<std::vector<int>>(prompt);
      promptInfo = "<" + std::to_string(tokens.size()) + " token ids>";
    }

    std::ostringstream out;
    out << "task_id=" << task_id << " model=" << model.value_or("default")
        << " stream=" << stream << " prompt=" << promptInfo
        << " max_tokens=" << detail::optStr(max_tokens)
        << " temperature=" << detail::optStr(temperature)
        << " top_p=" << detail::optStr(top_p)
        << " top_k=" << detail::optStr(top_k)
        << " min_p=" << detail::optStr(min_p)
        << " presence_penalty=" << presence_penalty
        << " frequency_penalty=" << frequency_penalty << " n=" << n
        << " stop_count=" << stop.size()
        << " sessionId=" << detail::optStr(sessionId)
        << " slotId=" << detail::optStr(slotId);
    return out.str();
  }
};

}  // namespace tt::domain
