// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <optional>
#include <string>
#include <json/json.h>

#include "utils/json_escape.hpp"

namespace tt::domain {

/**
 * Escape a string for safe embedding inside a JSON string literal.
 * Handles: \, ", and control characters U+0000–U+001F.
 */
inline std::string json_escape(const std::string& s) {
    std::string result;
    result.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  result.append("\\\""); break;
            case '\\': result.append("\\\\"); break;
            case '\b': result.append("\\b");  break;
            case '\f': result.append("\\f");  break;
            case '\n': result.append("\\n");  break;
            case '\r': result.append("\\r");  break;
            case '\t': result.append("\\t");  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    result.append(buf);
                } else {
                    result.push_back(static_cast<char>(c));
                }
        }
    }
    return result;
}

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
    std::optional<std::string> error;  // Error message if any

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

        return json;
    }

    std::string toJsonString() const {
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        return Json::writeString(writer, toJson());
    }

    // Fast SSE serialization - avoids Json::Value allocation overhead
    std::string toSSE() const {
        std::string result;
        result.reserve(256);  // Pre-allocate typical size

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
            result.append(tt::utils::json_escape(choices[i].text));
            result.append("\",\"index\":");
            result.append(std::to_string(choices[i].index));
            result.append(",\"logprobs\":null,\"finish_reason\":");
            if (choices[i].finish_reason.has_value()) {
                result.append("\"");
                result.append(choices[i].finish_reason.value());
                result.append("\"");
            } else {
                result.append("null");
            }
            result.append("}");
        }

        result.append("]}\n\n");
        return result;
    }
};

} // namespace tt::domain
