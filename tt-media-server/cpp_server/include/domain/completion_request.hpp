// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <json/json.h>

namespace tt::domain {

/**
 * Stream options for OpenAI-compatible streaming responses.
 */
struct StreamOptions {
    bool include_usage = true;
    bool continuous_usage_stats = false;

    static StreamOptions fromJson(const Json::Value& json) {
        StreamOptions opts;
        if (json.isMember("include_usage")) {
            opts.include_usage = json["include_usage"].asBool();
        }
        if (json.isMember("continuous_usage_stats")) {
            opts.continuous_usage_stats = json["continuous_usage_stats"].asBool();
        }
        return opts;
    }
};

/**
 * OpenAI-compatible completion request.
 * Based on OpenAI API specification.
 */
struct CompletionRequest {
    // Internal task tracking
    std::string task_id;

    // Model identifier
    std::optional<std::string> model;

    // Prompt can be a string or a list of token ids
    std::variant<std::string, std::vector<int>> prompt;

    // Response configuration
    bool echo = false;
    int max_tokens = 16;
    int n = 1;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    std::optional<std::string> suffix;
    bool stream = false;
    std::optional<StreamOptions> stream_options;

    // Stopping criteria
    std::vector<std::string> stop;

    // Reproducibility
    std::optional<int> seed;

    // Sampling params
    std::optional<float> temperature;
    std::optional<float> top_p;

    // Logging and debugging
    std::optional<int> logprobs;
    std::optional<std::string> user;

    // Completion sampling params
    bool use_beam_search = false;
    std::optional<int> top_k;
    std::optional<float> min_p;
    std::optional<float> repetition_penalty;
    float length_penalty = 1.0f;
    std::vector<int> stop_token_ids;
    bool include_stop_str_in_output = false;
    bool ignore_eos = false;
    int min_tokens = 0;
    bool skip_special_tokens = true;
    bool spaces_between_special_tokens = true;
    std::optional<std::vector<int>> allowed_token_ids;
    std::optional<int> prompt_logprobs;
    std::optional<int> truncate_prompt_tokens;

    static CompletionRequest fromJson(const Json::Value& json) {
        CompletionRequest req;

        if (json.isMember("model") && !json["model"].isNull()) {
            req.model = json["model"].asString();
        }

        if (json.isMember("prompt")) {
            if (json["prompt"].isString()) {
                req.prompt = json["prompt"].asString();
            } else if (json["prompt"].isArray()) {
                std::vector<int> tokens;
                for (const auto& token : json["prompt"]) {
                    tokens.push_back(token.asInt());
                }
                req.prompt = tokens;
            }
        }

        if (json.isMember("echo")) req.echo = json["echo"].asBool();
        if (json.isMember("max_tokens")) req.max_tokens = json["max_tokens"].asInt();
        if (json.isMember("n")) req.n = json["n"].asInt();
        if (json.isMember("presence_penalty")) req.presence_penalty = json["presence_penalty"].asFloat();
        if (json.isMember("frequency_penalty")) req.frequency_penalty = json["frequency_penalty"].asFloat();
        if (json.isMember("suffix") && !json["suffix"].isNull()) {
            req.suffix = json["suffix"].asString();
        }
        if (json.isMember("stream")) req.stream = json["stream"].asBool();
        if (json.isMember("stream_options") && !json["stream_options"].isNull()) {
            req.stream_options = StreamOptions::fromJson(json["stream_options"]);
        }

        if (json.isMember("stop")) {
            if (json["stop"].isString()) {
                req.stop.push_back(json["stop"].asString());
            } else if (json["stop"].isArray()) {
                for (const auto& s : json["stop"]) {
                    req.stop.push_back(s.asString());
                }
            }
        }

        if (json.isMember("seed") && !json["seed"].isNull()) {
            req.seed = json["seed"].asInt();
        }
        if (json.isMember("temperature") && !json["temperature"].isNull()) {
            req.temperature = json["temperature"].asFloat();
        }
        if (json.isMember("top_p") && !json["top_p"].isNull()) {
            req.top_p = json["top_p"].asFloat();
        }
        if (json.isMember("logprobs") && !json["logprobs"].isNull()) {
            req.logprobs = json["logprobs"].asInt();
        }
        if (json.isMember("user") && !json["user"].isNull()) {
            req.user = json["user"].asString();
        }

        if (json.isMember("use_beam_search")) req.use_beam_search = json["use_beam_search"].asBool();
        if (json.isMember("top_k") && !json["top_k"].isNull()) {
            req.top_k = json["top_k"].asInt();
        }
        if (json.isMember("min_p") && !json["min_p"].isNull()) {
            req.min_p = json["min_p"].asFloat();
        }
        if (json.isMember("repetition_penalty") && !json["repetition_penalty"].isNull()) {
            req.repetition_penalty = json["repetition_penalty"].asFloat();
        }
        if (json.isMember("length_penalty")) req.length_penalty = json["length_penalty"].asFloat();

        if (json.isMember("stop_token_ids") && json["stop_token_ids"].isArray()) {
            for (const auto& id : json["stop_token_ids"]) {
                req.stop_token_ids.push_back(id.asInt());
            }
        }

        if (json.isMember("include_stop_str_in_output")) {
            req.include_stop_str_in_output = json["include_stop_str_in_output"].asBool();
        }
        if (json.isMember("ignore_eos")) req.ignore_eos = json["ignore_eos"].asBool();
        if (json.isMember("min_tokens")) req.min_tokens = json["min_tokens"].asInt();
        if (json.isMember("skip_special_tokens")) req.skip_special_tokens = json["skip_special_tokens"].asBool();
        if (json.isMember("spaces_between_special_tokens")) {
            req.spaces_between_special_tokens = json["spaces_between_special_tokens"].asBool();
        }

        return req;
    }
};

} // namespace tt::domain
