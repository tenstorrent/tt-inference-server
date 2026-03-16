// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <json/json.h>

#include "domain/base_request.hpp"

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
struct CompletionRequest : BaseRequest {
    using BaseRequest::BaseRequest;

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
    int prompt_tokens_count = 0;

    static CompletionRequest fromJson(const Json::Value& json, TaskID task_id) {
        CompletionRequest req(std::move(task_id));

        if (json.isMember("model") && !json["model"].isNull()) {
            req.model = json["model"].asString();
        }

        if (json.isMember("prompt")) {
            if (json["prompt"].isString()) {
                req.prompt = json["prompt"].asString();
            } else if (json["prompt"].isArray()) {
                // Check if it's a batched request (array of prompts)
                if (json["prompt"].size() > 0 && json["prompt"][0].isArray()) {
                    // Batched requests are not currently supported
                    throw std::runtime_error(
                        "Batched completion requests (array of prompts) are not supported. "
                        "Please send one prompt per request. Received " +
                        std::to_string(json["prompt"].size()) + " prompts in batch.");
                } else {
                    // Flat array of token IDs for a single prompt
                    std::vector<int> tokens;
                    for (const auto& token : json["prompt"]) {
                        tokens.push_back(token.asInt());
                    }
                    req.prompt = tokens;
                }
            }
        }

        try {
            if (json.isMember("echo")) req.echo = json["echo"].asBool();
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse 'echo': ") + e.what());
        }
        try {
            if (json.isMember("max_tokens")) {
                // Support both int and string values for compatibility
                if (json["max_tokens"].isInt()) {
                    req.max_tokens = json["max_tokens"].asInt();
                } else if (json["max_tokens"].isString()) {
                    req.max_tokens = std::stoi(json["max_tokens"].asString());
                } else {
                    req.max_tokens = json["max_tokens"].asInt();  // Let it throw if wrong type
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse 'max_tokens': ") + e.what());
        }
        try {
            if (json.isMember("n")) {
                if (json["n"].isInt()) {
                    req.n = json["n"].asInt();
                } else if (json["n"].isString()) {
                    req.n = std::stoi(json["n"].asString());
                } else {
                    req.n = json["n"].asInt();
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse 'n': ") + e.what());
        }
        // Parse float fields with string fallback
        if (json.isMember("presence_penalty")) {
            if (json["presence_penalty"].isNumeric()) {
                req.presence_penalty = json["presence_penalty"].asFloat();
            } else if (json["presence_penalty"].isString()) {
                req.presence_penalty = std::stof(json["presence_penalty"].asString());
            }
        }
        if (json.isMember("frequency_penalty")) {
            if (json["frequency_penalty"].isNumeric()) {
                req.frequency_penalty = json["frequency_penalty"].asFloat();
            } else if (json["frequency_penalty"].isString()) {
                req.frequency_penalty = std::stof(json["frequency_penalty"].asString());
            }
        }
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

        try {
            if (json.isMember("seed") && !json["seed"].isNull()) {
                if (json["seed"].isInt()) {
                    req.seed = json["seed"].asInt();
                } else if (json["seed"].isString()) {
                    req.seed = std::stoi(json["seed"].asString());
                } else {
                    req.seed = json["seed"].asInt();
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse 'seed': ") + e.what());
        }
        if (json.isMember("temperature") && !json["temperature"].isNull()) {
            if (json["temperature"].isNumeric()) {
                req.temperature = json["temperature"].asFloat();
            } else if (json["temperature"].isString()) {
                req.temperature = std::stof(json["temperature"].asString());
            }
        }
        if (json.isMember("top_p") && !json["top_p"].isNull()) {
            if (json["top_p"].isNumeric()) {
                req.top_p = json["top_p"].asFloat();
            } else if (json["top_p"].isString()) {
                req.top_p = std::stof(json["top_p"].asString());
            }
        }
        try {
            if (json.isMember("logprobs") && !json["logprobs"].isNull()) {
                if (json["logprobs"].isInt()) {
                    req.logprobs = json["logprobs"].asInt();
                } else if (json["logprobs"].isString()) {
                    req.logprobs = std::stoi(json["logprobs"].asString());
                } else {
                    req.logprobs = json["logprobs"].asInt();
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse 'logprobs': ") + e.what());
        }
        if (json.isMember("user") && !json["user"].isNull()) {
            req.user = json["user"].asString();
        }

        if (json.isMember("use_beam_search")) req.use_beam_search = json["use_beam_search"].asBool();
        if (json.isMember("top_k") && !json["top_k"].isNull()) {
            req.top_k = json["top_k"].asInt();
        }
        if (json.isMember("min_p") && !json["min_p"].isNull()) {
            if (json["min_p"].isNumeric()) {
                req.min_p = json["min_p"].asFloat();
            } else if (json["min_p"].isString()) {
                req.min_p = std::stof(json["min_p"].asString());
            }
        }
        if (json.isMember("repetition_penalty") && !json["repetition_penalty"].isNull()) {
            if (json["repetition_penalty"].isNumeric()) {
                req.repetition_penalty = json["repetition_penalty"].asFloat();
            } else if (json["repetition_penalty"].isString()) {
                req.repetition_penalty = std::stof(json["repetition_penalty"].asString());
            }
        }
        if (json.isMember("length_penalty")) {
            if (json["length_penalty"].isNumeric()) {
                req.length_penalty = json["length_penalty"].asFloat();
            } else if (json["length_penalty"].isString()) {
                req.length_penalty = std::stof(json["length_penalty"].asString());
            }
        }

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
