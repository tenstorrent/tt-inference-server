// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <chrono>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iostream>
#include <thread>

#include "config/settings.hpp"
#include "runners/base_device_runner.hpp"
#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"
#include "utils/tokenizer.hpp"

namespace tt::runners {

// Simple logging helpers
struct LogStream {
    std::ostringstream ss;
    const char* level;
    LogStream(const char* l) : level(l) {}
    ~LogStream() { std::cout << "[" << level << "] " << ss.str() << std::endl; }
    template<typename T>
    LogStream& operator<<(const T& v) { ss << v; return *this; }
};

#define TT_LOG_INFO LogStream("INFO")
#define TT_LOG_DEBUG LogStream("DEBUG")

/**
 * Test runner for LLM streaming performance tests.
 * Generates fake tokens at an interval (ms) from TEST_RUNNER_FREQUENCY_MS env.
 * When a tokenizer is loaded (tokenizers/tokenizer.json next to exe), decodes fake token IDs to text (vLLM-style).
 */
class LLMTestRunner : public BaseDeviceRunner {
public:
    static constexpr int DEFAULT_TOKEN_INTERVAL_MS = 24;

    explicit LLMTestRunner(const std::string& device_id)
        : BaseDeviceRunner(device_id),
          token_interval_ms_(read_interval_from_env()),
          tokenizer_(tt::utils::TokenizerUtil(tt::config::tokenizer_path())) {
        TT_LOG_INFO << "LLMTestRunner initialized for device " << device_id
                 << ": token interval " << token_interval_ms_ << " ms";
    }

    bool warmup() override {
        TT_LOG_INFO << "LLMTestRunner warmup complete for device " << device_id_;
        return true;
    }

    std::vector<domain::CompletionResponse> run(
        const std::vector<domain::CompletionRequest>& requests) override {

        std::vector<domain::CompletionResponse> responses;
        responses.reserve(requests.size());

        for (const auto& request : requests) {
            domain::CompletionResponse response;
            response.id = "cmpl-" + request.task_id;
            response.created = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();
            response.model = request.model.value_or("test-model");

            // Generate all tokens as a single response (detokenize when tokenizer available)
            std::ostringstream text;
            std::vector<int> ids;
            ids.reserve(static_cast<size_t>(request.max_tokens));
            for (int i = 0; i < request.max_tokens; ++i) {
                ids.push_back(i);
            }
            auto decoded = tokenizer_.decode(ids);
            if (!decoded.empty()) {
                text << decoded;
            } else {
                for (int i = 0; i < request.max_tokens; ++i) {
                    text << "token_" << i << " ";
                }
            }

            domain::CompletionChoice choice;
            choice.text = text.str();
            choice.index = 0;
            choice.finish_reason = "stop";
            response.choices.push_back(choice);

            response.usage.completion_tokens = request.max_tokens;

            responses.push_back(response);
        }

        return responses;
    }

    void run_streaming(
        const domain::CompletionRequest& request,
        std::function<void(const domain::StreamingChunkOutput&)> chunk_callback,
        std::function<void(const domain::FinalResultOutput&)> final_callback) override {

        // Convert milliseconds to microseconds for sub-ms precision
        auto interval_us = std::chrono::microseconds(static_cast<long long>(token_interval_ms_ * 1000.0));
        auto next_token_time = std::chrono::steady_clock::now() + interval_us;

        for (int i = 0; i < request.max_tokens; ++i) {
            // Sleep until the next token time
            std::this_thread::sleep_until(next_token_time);
            next_token_time += interval_us;

            domain::StreamingChunkOutput chunk;
            chunk.task_id = request.task_id;
            auto decoded = tokenizer_.decode({i});
            if (!decoded.empty()) {
                chunk.chunk.text = decoded;
            } else {
                chunk.chunk.text = "token_" + std::to_string(i);
            }
            chunk.chunk.index = i;

            chunk_callback(chunk);
        }

        // Send final result
        domain::FinalResultOutput final_result;
        final_result.task_id = request.task_id;
        final_result.result.text = "";
        final_result.result.index = 0;
        final_result.result.finish_reason = "stop";
        final_result.return_result = true;

        final_callback(final_result);
    }

private:
    double token_interval_ms_;
    tt::utils::TokenizerUtil tokenizer_;

    static double read_interval_from_env() {
        const char* env = std::getenv("TEST_RUNNER_FREQUENCY_MS");
        if (env == nullptr) return DEFAULT_TOKEN_INTERVAL_MS;
        try {
            double val = std::stod(env);
            return (val > 0.0 && val <= 10000.0) ? val : DEFAULT_TOKEN_INTERVAL_MS;
        } catch (...) {
            return DEFAULT_TOKEN_INTERVAL_MS;
        }
    }
};

#undef TT_LOG_INFO
#undef TT_LOG_DEBUG

} // namespace tt::runners
