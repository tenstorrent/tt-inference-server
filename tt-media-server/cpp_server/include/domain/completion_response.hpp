// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <optional>
#include <string>

#include "domain/base_response.hpp"
#include "utils/json_escape.hpp"

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
 * A single choice in the completion response.
 */
struct CompletionChoice {
  std::string text;
  int index = 0;
  std::optional<Json::Value> logprobs;
  std::optional<std::string> finish_reason;
  std::optional<int64_t> token_id;
  std::optional<std::string> reasoning;  // Reasoning content for DeepSeek R1

  Json::Value toJson() const {
    Json::Value json;
    json["text"] = text;
    json["index"] = index;
    json["logprobs"] = logprobs.value_or(Json::nullValue);
    if (finish_reason.has_value()) {
      json["finish_reason"] = finish_reason.value();
    } else {
      json["finish_reason"] = Json::nullValue;
    }
    if (reasoning.has_value()) {
      json["reasoning"] = reasoning.value();
    }
    return json;
  }
};

/**
 * Full OpenAI-compatible completion response.
 */
struct CompletionResponse : BaseResponse {
  using BaseResponse::BaseResponse;

  std::string id;
  std::string object = "text_completion";
  int64_t created;
  std::string model;
  std::vector<CompletionChoice> choices;
  CompletionUsage usage;

  Json::Value toJson() const {
    Json::Value json;
    json["id"] = id;
    json["object"] = object;
    json["created"] = static_cast<Json::Int64>(created);
    json["model"] = model;

    Json::Value choicesArray(Json::arrayValue);
    for (const auto& choice : choices) {
      choicesArray.append(choice.toJson());
    }
    json["choices"] = choicesArray;
    json["usage"] = usage.toJson();

    return json;
  }

  std::string toJsonString() const {
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    writer["emitUTF8"] = true;
    return Json::writeString(writer, toJson());
  }
};

/**
 * Streaming chunk response (SSE format).
 */
struct StreamingChunkResponse : BaseResponse {
  using BaseResponse::BaseResponse;

  std::string id;
  std::string object = "text_completion";
  int64_t created;
  std::string model;
  std::vector<CompletionChoice> choices;
  std::optional<std::string> error;  // Error message if any
  std::optional<CompletionUsage> usage;

  Json::Value toJson() const {
    Json::Value json;
    json["id"] = id;
    json["object"] = object;
    json["created"] = static_cast<Json::Int64>(created);
    json["model"] = model;

    Json::Value choicesArray(Json::arrayValue);
    for (const auto& choice : choices) {
      choicesArray.append(choice.toJson());
    }
    json["choices"] = choicesArray;

    if (error.has_value()) {
      json["error"] = error.value();
    }

    if (usage.has_value()) {
      json["usage"] = usage->toJson();
    }

    return json;
  }

  std::string toJsonString() const {
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    writer["emitUTF8"] = true;
    return Json::writeString(writer, toJson());
  }

  // Fast SSE serialization - avoids Json::Value allocation overhead
  std::string toSSE() const {
    std::string result;
    result.reserve(512);

    result.append("data: {\"id\":\"");
    result.append(id);
    result.append("\",\"object\":\"");
    result.append(object);
    result.append("\",\"created\":");
    result.append(std::to_string(created));
    result.append(",\"model\":\"");
    result.append(model);
    result.append("\",\"choices\":[");

    for (size_t i = 0; i < choices.size(); ++i) {
      if (i > 0) result.append(",");
      result.append("{\"text\":\"");
      result.append(tt::utils::jsonEscape(choices[i].text));
      result.append("\",\"index\":");
      result.append(std::to_string(choices[i].index));
      result.append(",\"logprobs\":null");

      // Add reasoning field if present
      if (choices[i].reasoning.has_value()) {
        result.append(",\"reasoning\":\"");
        result.append(tt::utils::jsonEscape(choices[i].reasoning.value()));
        result.append("\"");
      }

      result.append(",\"finish_reason\":");
      if (choices[i].finish_reason.has_value()) {
        result.append("\"");
        result.append(choices[i].finish_reason.value());
        result.append("\"");
      } else {
        result.append("null");
      }
      result.append("}");
    }

    result.append("]");

    // Add usage data if present
    if (usage.has_value()) {
      result.append(",\"usage\":{\"prompt_tokens\":");
      result.append(std::to_string(usage->prompt_tokens));
      result.append(",\"completion_tokens\":");
      result.append(std::to_string(usage->completion_tokens));
      result.append(",\"total_tokens\":");
      result.append(std::to_string(usage->total_tokens));

      if (usage->ttft_ms.has_value()) {
        result.append(",\"ttft_ms\":");
        result.append(std::to_string(usage->ttft_ms.value()));
      }

      if (usage->tps.has_value()) {
        result.append(",\"tps\":");
        result.append(std::to_string(usage->tps.value()));
      }

      result.append("}");
    }

    result.append("}\n\n");
    return result;
  }
};

}  // namespace tt::domain
