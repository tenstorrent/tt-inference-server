// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <optional>
#include <string>

#include "domain/base_response.hpp"

namespace tt::domain::llm {

struct PromptTokensDetails {
  int cached_tokens = 0;
  int audio_tokens = 0;

  Json::Value toJson() const {
    Json::Value json;
    json["cached_tokens"] = cached_tokens;
    json["audio_tokens"] = audio_tokens;
    return json;
  }
};

struct CompletionTokensDetails {
  int reasoning_tokens = 0;
  int audio_tokens = 0;
  int accepted_prediction_tokens = 0;
  int rejected_prediction_tokens = 0;

  Json::Value toJson() const {
    Json::Value json;
    json["reasoning_tokens"] = reasoning_tokens;
    json["audio_tokens"] = audio_tokens;
    json["accepted_prediction_tokens"] = accepted_prediction_tokens;
    json["rejected_prediction_tokens"] = rejected_prediction_tokens;
    return json;
  }
};

/**
 * Usage statistics for the completion.
 */
struct CompletionUsage {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int total_tokens = 0;
  PromptTokensDetails prompt_tokens_details;
  CompletionTokensDetails completion_tokens_details;
  std::optional<double> ttft_ms;  // Time to first token in milliseconds
  std::optional<double> tps;      // Tokens per second (excluding first token)

  Json::Value toJson() const {
    Json::Value json;
    json["prompt_tokens"] = prompt_tokens;
    json["completion_tokens"] = completion_tokens;
    json["total_tokens"] = total_tokens;
    json["prompt_tokens_details"] = prompt_tokens_details.toJson();
    json["completion_tokens_details"] = completion_tokens_details.toJson();
    if (ttft_ms.has_value()) {
      json["ttft_ms"] = ttft_ms.value();
    }
    if (tps.has_value()) {
      json["tps"] = tps.value();
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
  uint32_t spec_accepts = 0;
  uint32_t spec_rejects = 0;
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

/**
 * Build a terminal error chunk carrying both `finish_reason="error"` on the
 * choice and a human-readable message in the chunk-level `error` field.
 */
inline LLMStreamChunk makeErrorChunk(uint32_t taskId, std::string error) {
  LLMStreamChunk chunk(taskId);
  LLMChoice choice;
  choice.finish_reason = "error";
  chunk.choices.push_back(std::move(choice));
  chunk.error = std::move(error);
  return chunk;
}

}  // namespace tt::domain::llm
