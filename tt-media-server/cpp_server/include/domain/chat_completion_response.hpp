// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <json/json.h>

#include <optional>
#include <string>
#include <vector>

#include "domain/llm_response.hpp"

namespace tt::domain {

/**
 * Message in a chat completion response choice.
 */
struct ChatCompletionMessage {
  std::string role = "assistant";
  std::string content;
  std::optional<std::string> reasoning;

  Json::Value toJson() const {
    Json::Value json;
    json["role"] = role;
    json["content"] = content;
    if (reasoning.has_value()) {
      json["reasoning"] = reasoning.value();
    }
    return json;
  }
};

/**
 * A single choice in the chat completion response.
 */
struct ChatCompletionChoice {
  int index = 0;
  ChatCompletionMessage message;
  std::optional<Json::Value> logprobs;
  std::string finish_reason = "stop";

  Json::Value toJson() const {
    Json::Value json;
    json["index"] = index;
    json["message"] = message.toJson();
    json["logprobs"] = logprobs.value_or(Json::nullValue);
    json["finish_reason"] = finish_reason;
    return json;
  }
};

/**
 * Full OpenAI-compatible chat completion response (non-streaming).
 */
struct ChatCompletionResponse {
  std::string id;
  std::string object = "chat.completion";
  int64_t created;
  std::string model;
  std::vector<ChatCompletionChoice> choices;
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

  static ChatCompletionResponse fromLLMResponse(const LLMResponse& completion) {
    ChatCompletionResponse response;
    response.id = completion.id;
    response.created = completion.created;
    response.model = completion.model;
    response.usage = completion.usage;

    for (const auto& choice : completion.choices) {
      ChatCompletionChoice chatChoice;
      chatChoice.index = choice.index;
      chatChoice.message.content = choice.text;
      chatChoice.message.reasoning = choice.reasoning;
      chatChoice.finish_reason = choice.finish_reason.value_or("stop");
      response.choices.push_back(std::move(chatChoice));
    }

    return response;
  }
};

/**
 * Delta content in a streaming chat completion chunk.
 */
struct ChatCompletionDelta {
  std::optional<std::string> role;
  std::optional<std::string> content;
  std::optional<std::string> reasoning;  // Reasoning content for DeepSeek R1

  Json::Value toJson() const {
    Json::Value json;
    if (role.has_value()) {
      json["role"] = role.value();
    }
    if (content.has_value()) {
      json["content"] = content.value();
    }
    if (reasoning.has_value()) {
      json["reasoning"] = reasoning.value();
    }
    return json;
  }
};

/**
 * A single choice in a streaming chat completion chunk.
 */
struct ChatCompletionStreamChoice {
  int index = 0;
  ChatCompletionDelta delta;
  std::optional<Json::Value> logprobs;
  std::optional<std::string> finish_reason;

  Json::Value toJson() const {
    Json::Value json;
    json["index"] = index;
    json["delta"] = delta.toJson();
    json["logprobs"] = logprobs.value_or(Json::nullValue);
    if (finish_reason.has_value()) {
      json["finish_reason"] = finish_reason.value();
    } else {
      json["finish_reason"] = Json::nullValue;
    }
    return json;
  }
};

/**
 * Streaming chat completion chunk response (SSE format).
 */
struct ChatCompletionStreamChunk {
  std::string id;
  std::string object = "chat.completion.chunk";
  int64_t created;
  std::string model;
  std::vector<ChatCompletionStreamChoice> choices;
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

  std::string toSSE() const {
    std::string result;
    result.reserve(256);
    result.append("data: ");
    result.append(toJsonString());
    result.append("\n\n");
    return result;
  }

  /**
   * Create an initial role-only chunk (sent before content generation starts).
   * vLLM convention: delta has role="assistant" and empty content,
   * finish_reason is null.
   */
  static ChatCompletionStreamChunk makeInitialChunk(
      const std::string& id, const std::string& model, int64_t created,
      const std::optional<CompletionUsage>& usage = std::nullopt) {
    ChatCompletionStreamChunk chunk;
    chunk.id = id;
    chunk.created = created;
    chunk.model = model;
    chunk.usage = usage;

    ChatCompletionStreamChoice choice;
    choice.delta.role = "assistant";
    choice.delta.content = "";
    chunk.choices.push_back(std::move(choice));

    return chunk;
  }

  /**
   * Create a content delta chunk from an LLMChoice.
   */
  static ChatCompletionStreamChunk makeContentChunk(
      const std::string& id, const std::string& model, int64_t created,
      const LLMChoice& completionChoice,
      const std::optional<CompletionUsage>& usage = std::nullopt) {
    ChatCompletionStreamChunk chunk;
    chunk.id = id;
    chunk.created = created;
    chunk.model = model;
    chunk.usage = usage;

    ChatCompletionStreamChoice choice;
    choice.index = completionChoice.index;
    choice.delta.content = completionChoice.text;

    // Include reasoning content if present (DeepSeek R1 style)
    if (completionChoice.reasoning.has_value()) {
      choice.delta.reasoning = completionChoice.reasoning;
    }

    if (completionChoice.finish_reason.has_value()) {
      const std::string& reason = completionChoice.finish_reason.value();
      choice.finish_reason = reason.empty() ? "stop" : reason;
    }

    chunk.choices.push_back(std::move(choice));

    return chunk;
  }

  /**
   * Create a final usage-only chunk (empty choices, usage stats).
   * Sent when stream_options.include_usage is true.
   */
  static ChatCompletionStreamChunk makeUsageChunk(
      const std::string& id, const std::string& model, int64_t created,
      const CompletionUsage& usage) {
    ChatCompletionStreamChunk chunk;
    chunk.id = id;
    chunk.created = created;
    chunk.model = model;
    chunk.usage = usage;
    // Empty choices array
    return chunk;
  }
};

}  // namespace tt::domain
