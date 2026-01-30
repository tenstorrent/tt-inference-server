// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <string>
#include <optional>
#include <json/json.h>

namespace tt::domain {

/**
 * Represents a single streaming chunk from the completion.
 */
struct CompletionStreamChunk {
    std::string text;
    std::optional<int> index;
    std::optional<std::string> finish_reason;

    Json::Value toJson() const {
        Json::Value json;
        json["text"] = text;
        if (index.has_value()) {
            json["index"] = index.value();
        } else {
            json["index"] = Json::nullValue;
        }
        if (finish_reason.has_value()) {
            json["finish_reason"] = finish_reason.value();
        } else {
            json["finish_reason"] = Json::nullValue;
        }
        return json;
    }
};

/**
 * Output yielded during streaming generation.
 */
struct StreamingChunkOutput {
    static constexpr const char* TYPE = "streaming_chunk";
    CompletionStreamChunk chunk;
    std::string task_id;
};

/**
 * Final output yielded at the end of streaming generation.
 */
struct FinalResultOutput {
    static constexpr const char* TYPE = "final_result";
    CompletionStreamChunk result;
    std::string task_id;
    bool return_result;
};

/**
 * Usage statistics for the completion.
 */
struct CompletionUsage {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;

    Json::Value toJson() const {
        Json::Value json;
        json["prompt_tokens"] = prompt_tokens;
        json["completion_tokens"] = completion_tokens;
        json["total_tokens"] = total_tokens;
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
        return json;
    }
};

/**
 * Full OpenAI-compatible completion response.
 */
struct CompletionResponse {
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
        return Json::writeString(writer, toJson());
    }
};

/**
 * Streaming chunk response (SSE format).
 */
struct StreamingChunkResponse {
    std::string id;
    std::string object = "text_completion";
    int64_t created;
    std::string model;
    std::vector<CompletionChoice> choices;

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

        return json;
    }

    std::string toJsonString() const {
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        return Json::writeString(writer, toJson());
    }

    std::string toSSE() const {
        return "data: " + toJsonString() + "\n\n";
    }
};

} // namespace tt::domain
