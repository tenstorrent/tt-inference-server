// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <chrono>
#include <string>
#include <sstream>
#include <iostream>

#include "runners/base_device_runner.hpp"
#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"

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
 * Generates fake tokens at 120,000 tokens per second to test the streaming
 * infrastructure without requiring actual model inference.
 *
 * At 120,000 tokens/second:
 *   - Token interval: ~8.33 microseconds
 *   - This is significantly faster than real LLM inference
 *   - Used for benchmarking server overhead
 */
class LLMTestRunner : public BaseDeviceRunner {
public:
    // Target: 120,000 tokens per second
    static constexpr double TOKENS_PER_SECOND = 120000.0;
    static constexpr double MICROSECONDS_PER_SECOND = 1000000.0;
    static constexpr double TOKEN_INTERVAL_MICROSECONDS = MICROSECONDS_PER_SECOND / TOKENS_PER_SECOND;

    explicit LLMTestRunner(const std::string& device_id)
        : BaseDeviceRunner(device_id)
        , token_interval_us_(TOKEN_INTERVAL_MICROSECONDS) {
        TT_LOG_INFO << "LLMTestRunner initialized for device " << device_id
                 << ": target " << TOKENS_PER_SECOND << " tokens/sec"
                 << " (interval: " << token_interval_us_ << " µs)";
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

            // Generate all tokens as a single response
            std::ostringstream text;
            for (int i = 0; i < request.max_tokens; ++i) {
                text << "token_" << i << " ";
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

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < request.max_tokens; ++i) {
            // Calculate exact target time for this token
            auto target_time = start_time + std::chrono::microseconds(
                static_cast<int64_t>(i * token_interval_us_)
            );

            // Busy-wait for precise timing (sleep is too coarse for µs precision)
            while (std::chrono::high_resolution_clock::now() < target_time) {
                // Spin - we need microsecond precision
                // Could use _mm_pause() on x86 for better power efficiency
            }

            // Create and emit chunk
            domain::StreamingChunkOutput chunk;
            chunk.task_id = request.task_id;
            chunk.chunk.text = "token_" + std::to_string(i);
            chunk.chunk.index = i;

            chunk_callback(chunk);
        }

        // Send final result
        domain::FinalResultOutput final_result;
        final_result.task_id = request.task_id;
        final_result.result.text = "[DONE]";
        final_result.result.index = 0;
        final_result.result.finish_reason = "stop";
        final_result.return_result = true;

        final_callback(final_result);

        // Log actual performance
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time
        ).count();

        double actual_tokens_per_sec = (request.max_tokens * MICROSECONDS_PER_SECOND) / duration_us;
        TT_LOG_DEBUG << "LLMTestRunner generated " << request.max_tokens
                  << " tokens in " << duration_us << " µs"
                  << " (" << actual_tokens_per_sec << " tokens/sec)";
    }

private:
    double token_interval_us_;
};

#undef TT_LOG_INFO
#undef TT_LOG_DEBUG

} // namespace tt::runners
