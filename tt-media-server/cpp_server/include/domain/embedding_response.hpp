// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <vector>
#include <string>
#include <json/json.h>

#include "domain/base_response.hpp"

namespace tt::domain {

/**
 * OpenAI-compatible embedding response.
 * Based on: https://platform.openai.com/docs/api-reference/embeddings/object
 */
struct EmbeddingResponse : BaseResponse {
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
    Json::Value to_openai_json() const {
        Json::Value response;
        response["object"] = "list";

        // Data array with embedding object
        Json::Value data(Json::arrayValue);
        Json::Value embedding_obj;
        embedding_obj["object"] = "embedding";
        embedding_obj["index"] = 0;

        // Pre-size the array for better performance
        Json::Value embedding_array(Json::arrayValue);
        embedding_array.resize(static_cast<Json::ArrayIndex>(embedding.size()));
        for (size_t i = 0; i < embedding.size(); ++i) {
            embedding_array[static_cast<Json::ArrayIndex>(i)] = embedding[i];
        }
        embedding_obj["embedding"] = std::move(embedding_array);
        data.append(std::move(embedding_obj));

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
     * Parse from Python result JSON.
     */
    static EmbeddingResponse from_json(const Json::Value& json) {
        EmbeddingResponse resp;

        if (json.isMember("embedding") && json["embedding"].isArray()) {
            const Json::Value& emb_array = json["embedding"];
            const size_t size = emb_array.size();
            resp.embedding.reserve(size);
            for (Json::ArrayIndex i = 0; i < size; ++i) {
                resp.embedding.push_back(emb_array[i].asFloat());
            }
        }

        if (json.isMember("total_tokens")) {
            resp.total_tokens = json["total_tokens"].asInt();
        }

        if (json.isMember("model")) {
            resp.model = json["model"].asString();
        }

        if (json.isMember("task_id")) {
            resp.task_id = TaskID(json["task_id"].asString());
        }

        if (json.isMember("error")) {
            resp.error = json["error"].asString();
        }

        return resp;
    }
};

} // namespace tt::domain
