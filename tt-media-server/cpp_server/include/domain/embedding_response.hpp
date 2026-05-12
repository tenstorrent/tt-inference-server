// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <string>
#include <vector>

#include "domain/base_response.hpp"
#include "utils/id_generator.hpp"

namespace tt::domain {

/**
 * OpenAI-compatible embedding response.
 * Based on: https://platform.openai.com/docs/api-reference/embeddings/object
 */
struct EmbeddingResponse : BaseResponse {
  using BaseResponse::BaseResponse;

  // The embedding vector
  std::vector<float> embedding;

  // Number of tokens in the input
  int total_tokens = 0;

  // Model used
  std::string model;

  // Error message (if any)
  std::string error;

  /**
   * Convert to OpenAI-compatible JSON response.
   */
  Json::Value toOpenaiJson() const {
    Json::Value response;
    response["object"] = "list";

    // Data array with embedding object
    Json::Value data(Json::arrayValue);
    Json::Value embeddingObj;
    embeddingObj["object"] = "embedding";
    embeddingObj["index"] = 0;

    // Pre-size the array for better performance
    Json::Value embeddingArray(Json::arrayValue);
    embeddingArray.resize(static_cast<Json::ArrayIndex>(embedding.size()));
    for (size_t i = 0; i < embedding.size(); ++i) {
      embeddingArray[static_cast<Json::ArrayIndex>(i)] = embedding[i];
    }
    embeddingObj["embedding"] = std::move(embeddingArray);
    data.append(std::move(embeddingObj));

    response["data"] = std::move(data);
    response["model"] = model;

    // Usage info
    Json::Value usage;
    usage["total_tokens"] = total_tokens;
    usage["prompt_tokens"] = total_tokens;
    response["usage"] = std::move(usage);

    return response;
  }

  /**
   * Parse from Python result JSON. task_id from JSON if present, otherwise a
   * new UUID.
   */
  static EmbeddingResponse fromJson(const Json::Value& json) {
    uint32_t tid = json.isMember("task_id") && json["task_id"].isUInt()
                       ? json["task_id"].asUInt()
                       : tt::utils::TaskIDGenerator::generate();
    EmbeddingResponse resp(tid);

    if (json.isMember("embedding") && json["embedding"].isArray()) {
      const Json::Value& embArray = json["embedding"];
      const size_t SIZE = embArray.size();
      resp.embedding.reserve(SIZE);
      for (Json::ArrayIndex i = 0; i < SIZE; ++i) {
        resp.embedding.push_back(embArray[i].asFloat());
      }
    }

    if (json.isMember("total_tokens")) {
      resp.total_tokens = json["total_tokens"].asInt();
    }

    if (json.isMember("model")) {
      resp.model = json["model"].asString();
    }

    if (json.isMember("error")) {
      resp.error = json["error"].asString();
    }

    return resp;
  }
};

}  // namespace tt::domain
