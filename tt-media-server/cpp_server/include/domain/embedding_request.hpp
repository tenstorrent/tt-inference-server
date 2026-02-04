// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <string>
#include <optional>
#include <json/json.h>

namespace tt::domain {

/**
 * OpenAI-compatible embedding request.
 * Based on: https://platform.openai.com/docs/api-reference/embeddings/create
 */
struct EmbeddingRequest {
    // Required: Model to use for embedding
    std::string model;

    // Required: Text to embed
    std::string input;

    // Optional: User identifier
    std::optional<std::string> user;

    // Internal: Task ID for tracking
    std::string task_id;

    /**
     * Parse from JSON.
     */
    static EmbeddingRequest from_json(const Json::Value& json) {
        EmbeddingRequest req;

        if (json.isMember("model")) {
            req.model = json["model"].asString();
        }

        if (json.isMember("input")) {
            req.input = json["input"].asString();
        }

        if (json.isMember("user")) {
            req.user = json["user"].asString();
        }

        if (json.isMember("task_id")) {
            req.task_id = json["task_id"].asString();
        }

        return req;
    }

    /**
     * Convert to JSON for IPC.
     */
    Json::Value to_json() const {
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

} // namespace tt::domain
