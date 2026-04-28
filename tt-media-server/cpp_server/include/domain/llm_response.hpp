// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <optional>
#include <string>

#include "domain/base_response.hpp"

namespace tt::domain {

/**
 * Usage statistics for the completion.
 */
struct CompletionUsage {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int total_tokens = 0;
  std::optional<double> ttft_ms;  // Time to first token in milliseconds
  std::optional<double> tps;      // Tokens per second (excluding first token)
  std::optional<std::string>
      sessionId;  // Session ID if session management is used

  Json::Value toJson() const {
    Json::Value json;
    json["prompt_tokens"] = prompt_tokens;
    json["completion_tokens"] = completion_tokens;
    json["total_tokens"] = total_tokens;
    if (ttft_ms.has_value()) {
      json["ttft_ms"] = ttft_ms.value();
    }
    if (tps.has_value()) {
      json["tps"] = tps.value();
    }
    if (sessionId.has_value()) {
      json["sessionId"] = sessionId.value();
    }
    return json;
  }
};

/**
 * A single choice in the LLM pipeline response.
 */
struct LLMChoice {
  std::string text;
  int index = 0;
  std::optional<Json::Value> logprobs;
  std::optional<std::string> finish_reason;
  std::optional<int64_t> token_id;
  std::optional<std::string> reasoning;
  std::optional<Json::Value> tool_calls;
};

/**
 * Internal LLM pipeline response.
 * Converted to ChatCompletionResponse before being sent to clients.
 */
struct LLMResponse : BaseResponse {
  using BaseResponse::BaseResponse;

  std::string id;
  std::string object = "text_completion";
  int64_t created;
  std::string model;
  std::vector<LLMChoice> choices;
  CompletionUsage usage;
};

/**
 * Internal streaming chunk used as the callback type within LLMService.
 * Converted to ChatCompletionStreamChunk before being sent to clients.
 */
struct LLMStreamChunk : BaseResponse {
  using BaseResponse::BaseResponse;

  std::string id;
  std::string object = "text_completion";
  int64_t created;
  std::string model;
  std::vector<LLMChoice> choices;
  std::optional<std::string> error;
  std::optional<CompletionUsage> usage;
};

}  // namespace tt::domain
