// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <chrono>
#include <string>
#include <sstream>
#include <iostream>
#include <thread>

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
 * Generates fake tokens with a 24ms interval to simulate real model inference
 * timing and allow meaningful ITL benchmarking.
 */
class LLMTestRunner : public BaseDeviceRunner {
public:
    static constexpr int TOKEN_INTERVAL_MS = 24;

    explicit LLMTestRunner(const std::string& device_id)
        : BaseDeviceRunner(device_id) {
        TT_LOG_INFO << "LLMTestRunner initialized for device " << device_id
                 << ": token interval " << TOKEN_INTERVAL_MS << " ms";
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

        for (int i = 0; i < request.max_tokens; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(TOKEN_INTERVAL_MS));

            domain::StreamingChunkOutput chunk;
            chunk.task_id = request.task_id;
            chunk.chunk.text = "tok";
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
    }

private:
};

#undef TT_LOG_INFO
#undef TT_LOG_DEBUG

} // namespace tt::runners
