// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <optional>
#include <string>

#include "domain/base_request.hpp"
#include "domain/json_field.hpp"

namespace tt::domain {

/**
 * OpenAI-compatible embedding request.
 * Based on: https://platform.openai.com/docs/api-reference/embeddings/create
 */
struct EmbeddingRequest : BaseRequest {
  using BaseRequest::BaseRequest;

  // Required: Model to use for embedding
  std::string model;

  // Required: Text to embed
  std::string input;

  // Optional: User identifier
  std::optional<std::string> user;

  /**
   * Parse from JSON. task_id must be provided (e.g. from controller).
   */
  static EmbeddingRequest fromJson(const Json::Value& json, uint32_t taskId) {
    EmbeddingRequest req(std::move(taskId));
    if (json.isMember("model"))
      req.model = json_field::getString(json["model"], "model");
    if (json.isMember("input"))
      req.input = json_field::getString(json["input"], "input");
    if (json.isMember("user"))
      req.user = json_field::getString(json["user"], "user");
    return req;
  }

  /**
   * Convert to JSON for IPC.
   */
  Json::Value toJson() const {
    Json::Value json;
    json["model"] = model;
    json["input"] = input;
    json["task_id"] = task_id;
    if (user) {
      json["user"] = *user;
    }
    return json;
  }
};

}  // namespace tt::domain
