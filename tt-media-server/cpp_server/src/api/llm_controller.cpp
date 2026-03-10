// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "api/llm_controller.hpp"
#include "config/settings.hpp"
#include "domain/chat_completion_request.hpp"
#include "domain/chat_completion_response.hpp"
#include "domain/completion_response.hpp"
#include "profiling/tracy.hpp"
#include "utils/logger.hpp"

#include <memory>
#include <random>
#include <sstream>
#include <chrono>
#include <iostream>
#include <cmath>
#include <mutex>

#include "utils/service_factory.hpp"
#include <json/json.h>
#include <trantor/net/EventLoop.h>

namespace tt::api {

LLMController::LLMController() {
    if (!tt::config::is_llm_service_enabled()) {
        TT_LOG_INFO("[LLMController] Skipping initialization (TT_MODEL_SERVICE != llm)");
        return;
    }

    service_ = tt::utils::service_factory::get_service_by_type<services::LLMService>();
    if (!service_) {
        throw std::runtime_error("[LLMController] LLM service not found in service fabric. "
                                 "Ensure register_services() is called before Drogon starts.");
    }
    TT_LOG_INFO("[LLMController] Initialized (service already started)");
}

std::string LLMController::generate_completion_id() {
    static std::mutex gen_mutex;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static const char* hex_chars = "0123456789abcdef";

    std::lock_guard<std::mutex> lock(gen_mutex);
    std::ostringstream ss;
    for (int i = 0; i < 24; ++i) {
        ss << hex_chars[dis(gen)];
    }
    return ss.str();
}

Json::Value LLMController::error_json(const std::string& message, const std::string& type,
    const Json::Value& param, const Json::Value& code) {
    Json::Value error;
    error["object"] = "error";
    error["message"] = message;
    error["type"] = type;
    error["param"] = param;
    error["code"] = code;
    return error;
}

void LLMController::completions(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {

    ZoneScopedN("API::completions");

    auto json = req->getJsonObject();
    if (!json) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            error_json("Invalid JSON body", "invalid_request_error"));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    auto request = std::make_shared<domain::CompletionRequest>();
    try {
        *request = domain::CompletionRequest::fromJson(*json);
        request->task_id = domain::TaskID(generate_completion_id());
    } catch (const std::exception& e) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            error_json(std::string("Failed to parse request: ") + e.what(), "invalid_request_error"));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    if (!service_->is_model_ready()) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            error_json("Model is not ready", "service_unavailable"));
        resp->setStatusCode(drogon::k503ServiceUnavailable);
        callback(resp);
        return;
    }

    if (request->stream) {
        handle_streaming(request, std::move(callback), false);
    } else {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto response = service_->submit_request(std::move(*request));
        auto end_time = std::chrono::high_resolution_clock::now();

        response.id = "cmpl-" + response.id;

        // Add timing metrics to non-streaming response
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        if (total_duration.count() > 0 && response.usage.completion_tokens > 0) {
            response.usage.ttft_ms = static_cast<double>(total_duration.count());
            if (response.usage.completion_tokens > 1) {
                response.usage.tps = response.usage.completion_tokens * 1000.0 / total_duration.count();
            }
        }

        callback(drogon::HttpResponse::newHttpJsonResponse(response.toJson()));
    }
}

void LLMController::chat_completions(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {
    ZoneScopedN("API::chat_completions");

    auto json = req->getJsonObject();
    if (!json) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            error_json("Invalid JSON body", "invalid_request_error"));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    domain::ChatCompletionRequest chat_req;
    try {
        chat_req = domain::ChatCompletionRequest::fromJson(*json);
        chat_req.task_id = domain::TaskID(generate_completion_id());
    } catch (const std::exception& e) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            error_json(std::string("Failed to parse request: ") + e.what(), "invalid_request_error"));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    if (chat_req.messages.empty()) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            error_json("messages is required and must be a non-empty array",
                "invalid_request_error", Json::Value("messages")));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    if (!service_->is_model_ready()) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            error_json("Model is not ready", "service_unavailable"));
        resp->setStatusCode(drogon::k503ServiceUnavailable);
        callback(resp);
        return;
    }

    auto request = std::make_shared<domain::CompletionRequest>(chat_req.to_completion_request());

    if (request->stream) {
        handle_streaming(request, std::move(callback), true);
    } else {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto completion = service_->submit_request(std::move(*request));
        auto end_time = std::chrono::high_resolution_clock::now();

        completion.id = "chatcmpl-" + completion.id;

        // Add timing metrics to non-streaming response
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        if (total_duration.count() > 0 && completion.usage.completion_tokens > 0) {
            completion.usage.ttft_ms = static_cast<double>(total_duration.count());
            if (completion.usage.completion_tokens > 1) {
                completion.usage.tps = completion.usage.completion_tokens * 1000.0 / total_duration.count();
            }
        }

        auto chat_response = domain::ChatCompletionResponse::fromCompletionResponse(completion);
        callback(drogon::HttpResponse::newHttpJsonResponse(chat_response.toJson()));
    }
}

void LLMController::handle_streaming(
    std::shared_ptr<domain::CompletionRequest> req_ptr,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback,
    bool is_chat) const {

    ZoneScopedN("API::handle_streaming");

    const std::string completion_id =
        (is_chat ? "chatcmpl-" : "cmpl-") + req_ptr->task_id.id;
    const std::string model = req_ptr->model.value_or("default");
    const int64_t created = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    const bool include_usage = !req_ptr->stream_options.has_value()
        || req_ptr->stream_options->include_usage;  // Default to true if not specified
    const bool continuous_usage = req_ptr->stream_options.has_value()
        && req_ptr->stream_options->continuous_usage_stats;

    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [this, req_ptr, completion_id, model, created,
         is_chat, include_usage, continuous_usage](
            drogon::ResponseStreamPtr stream) mutable {
            trantor::EventLoop* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
            auto done = std::make_shared<std::atomic<bool>>(false);
            auto stream_ptr =
                std::make_shared<drogon::ResponseStreamPtr>(std::move(stream));
            auto completion_tokens = std::make_shared<std::atomic<int>>(0);

            // Timing metrics for TTFT and TPS calculation
            auto start_time = std::make_shared<std::chrono::high_resolution_clock::time_point>(
                std::chrono::high_resolution_clock::now());
            auto first_token_time = std::make_shared<std::optional<std::chrono::high_resolution_clock::time_point>>();
            auto second_token_time = std::make_shared<std::optional<std::chrono::high_resolution_clock::time_point>>();
            auto first_content_chunk = std::make_shared<std::atomic<bool>>(true);

            service_->submit_streaming_request(
                *req_ptr,
                [loop, stream_ptr, done, completion_id, model, created,
                 is_chat, include_usage, continuous_usage, completion_tokens,
                 start_time, first_token_time, second_token_time, first_content_chunk, req_ptr](
                    const domain::StreamingChunkResponse& chunk, bool is_final) {
                    if (done->load() || !*stream_ptr) {
                        return;
                    }
                    if (!chunk.choices.empty()) {
                        const int current_tokens = completion_tokens->fetch_add(1) + 1;

                        // Record timing for TTFT and TPS calculation
                        auto now = std::chrono::high_resolution_clock::now();
                        if (!first_token_time->has_value()) {
                            *first_token_time = now;
                        } else if (current_tokens == 2 && !second_token_time->has_value()) {
                            *second_token_time = now;
                        }

                        std::string sse;
                        if (is_chat) {
                            std::optional<domain::CompletionUsage> usage;
                            if (continuous_usage) {
                                // Only send token counts during streaming, timing metrics come with final chunk
                                usage = domain::CompletionUsage{req_ptr->prompt_tokens_count, current_tokens, current_tokens, std::nullopt, std::nullopt};
                            }
                            auto stream_chunk = domain::ChatCompletionStreamChunk::makeContentChunk(
                                completion_id, model, created, chunk.choices[0], usage);
                            if (first_content_chunk->exchange(false)) {
                                std::optional<domain::CompletionUsage> initial_usage;
                                if (continuous_usage) {
                                    initial_usage = domain::CompletionUsage{req_ptr->prompt_tokens_count, 0, 0, std::nullopt, std::nullopt};
                                }
                                auto initial_chunk = domain::ChatCompletionStreamChunk::makeInitialChunk(
                                    completion_id, model, created, initial_usage);
                                sse = initial_chunk.toSSE() + stream_chunk.toSSE();
                            } else {
                                sse = stream_chunk.toSSE();
                            }
                        } else if (!chunk.choices[0].text.empty() ||
                                   !chunk.choices[0].finish_reason.has_value()) {
                            domain::StreamingChunkResponse out;
                            out.id = completion_id;
                            out.object = "text_completion";
                            out.model = model;
                            out.created = created;
                            out.choices = chunk.choices;
                            sse = out.toSSE();
                        }

                        if (!sse.empty()) {
                            loop->queueInLoop(
                                [stream_ptr, sse = std::move(sse)]() {
                                if (*stream_ptr) (*stream_ptr)->send(sse);
                            });
                        }
                    }
                    if (is_final) {
                        loop->queueInLoop(

                            [stream_ptr, done, is_chat, include_usage,
                             completion_id, model, created, completion_tokens,
                             start_time, first_token_time, second_token_time, req_ptr]() {
                                if (!done->exchange(true) && *stream_ptr) {
                                    if (include_usage) {
                                        const int tokens = completion_tokens->load();
                                        domain::CompletionUsage usage{req_ptr->prompt_tokens_count, tokens, tokens, std::nullopt, std::nullopt};

                                        // Calculate final timing metrics
                                        if (first_token_time->has_value()) {
                                            auto ttft_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                                                first_token_time->value() - *start_time);
                                            usage.ttft_ms = std::round(static_cast<double>(ttft_duration.count()) / 10.0) / 100.0;
                                        }

                                        if (tokens > 1) {
                                            auto final_time = std::chrono::high_resolution_clock::now();
                                            auto base_time = second_token_time->has_value() ?
                                                second_token_time->value() : first_token_time->value();
                                            auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                                                final_time - base_time);
                                            if (total_duration.count() > 0) {
                                                auto time_seconds = static_cast<double>(total_duration.count()) / 1000000.0;
                                                usage.tps = std::round((tokens - 1) / time_seconds * 1000.0) / 1000.0;
                                                TT_LOG_DEBUG("[LLMController] Final TPS: {} tokens/sec", usage.tps.value());
                                            }
                                        }

                                        if (is_chat) {
                                            (*stream_ptr)->send(
                                                domain::ChatCompletionStreamChunk::makeUsageChunk(
                                                    completion_id, model, created, usage).toSSE());
                                        } else {
                                            // For text completions, we need to send a usage chunk
                                            // The format should be similar but for text_completion object
                                            domain::StreamingChunkResponse usage_chunk;
                                            usage_chunk.id = completion_id;
                                            usage_chunk.object = "text_completion";
                                            usage_chunk.model = model;
                                            usage_chunk.created = created;
                                            usage_chunk.usage = usage;
                                            (*stream_ptr)->send(usage_chunk.toSSE());
                                        }
                                    }
                                    (*stream_ptr)->send("data: [DONE]\n\n");
                                    (*stream_ptr)->close();
                                }
                            });
                    }
                });
        });

    resp->setContentTypeString("text/event-stream");
    resp->addHeader("Cache-Control", "no-cache");
    resp->addHeader("Connection", "keep-alive");
    resp->addHeader("X-Accel-Buffering", "no");

    callback(resp);
}

} // namespace tt::api
