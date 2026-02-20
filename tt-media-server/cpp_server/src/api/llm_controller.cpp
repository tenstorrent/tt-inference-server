// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "api/llm_controller.hpp"
#include "config/settings.hpp"
#include "domain/chat_completion_request.hpp"
#include "domain/chat_completion_response.hpp"
#include "domain/completion_response.hpp"
#include "profiling/tracy.hpp"

#include <memory>
#include <random>
#include <sstream>
#include <chrono>
#include <iostream>
#include <mutex>

#include "utils/service_factory.hpp"
#include <json/json.h>
#include <trantor/net/EventLoop.h>

namespace tt::api {

LLMController::LLMController() {
    if (!tt::config::is_llm_service_enabled()) {
        std::cout << "[LLMController] Skipping initialization (TT_MODEL_SERVICE != llm)" << std::endl;
        return;
    }

    service_ = tt::utils::service_factory::get_service<services::LLMService>();
    if (!service_) {
        throw std::runtime_error("[LLMController] LLM service not found in service fabric. "
                                 "Ensure register_services() is called before Drogon starts.");
    }
    std::cout << "[LLMController] Initialized (service already started)" << std::endl;
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

    domain::CompletionRequest request;
    try {
        request = domain::CompletionRequest::fromJson(*json);
        request.task_id = generate_completion_id();
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

    if (request.stream) {
        handle_streaming(std::move(request), std::move(callback), false);
    } else {
        auto response = service_->submit_request(std::move(request));
        response.id = "cmpl-" + response.id;
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
        chat_req.task_id = generate_completion_id();
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

    domain::CompletionRequest request = chat_req.to_completion_request();

    if (request.stream) {
        handle_streaming(std::move(request), std::move(callback), true);
    } else {
        auto completion = service_->submit_request(std::move(request));
        completion.id = "chatcmpl-" + completion.id;
        auto chat_response =
            domain::ChatCompletionResponse::fromCompletionResponse(completion);
        callback(drogon::HttpResponse::newHttpJsonResponse(chat_response.toJson()));
    }
}

void LLMController::handle_streaming(
    domain::CompletionRequest request,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback,
    bool is_chat) const {

    ZoneScopedN("API::handle_streaming");

    const std::string completion_id =
        (is_chat ? "chatcmpl-" : "cmpl-") + request.task_id;
    const std::string model = request.model.value_or("default");
    const int64_t created = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());

    const bool include_usage = is_chat && request.stream_options.has_value()
        && request.stream_options->include_usage;
    const bool continuous_usage = is_chat && request.stream_options.has_value()
        && request.stream_options->continuous_usage_stats;

    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [this, req = std::move(request), completion_id, model, created,
         is_chat, include_usage, continuous_usage](
            drogon::ResponseStreamPtr stream) mutable {
            trantor::EventLoop* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
            auto done = std::make_shared<std::atomic<bool>>(false);
            auto stream_ptr =
                std::make_shared<drogon::ResponseStreamPtr>(std::move(stream));
            auto completion_tokens = std::make_shared<std::atomic<int>>(0);

            if (is_chat) {
                std::optional<domain::CompletionUsage> initial_usage;
                if (continuous_usage) {
                    initial_usage = domain::CompletionUsage{0, 0, 0};
                }
                auto initial_chunk = domain::ChatCompletionStreamChunk::makeInitialChunk(
                    completion_id, model, created, initial_usage);
                if (*stream_ptr) (*stream_ptr)->send(initial_chunk.toSSE());
            }

            service_->submit_streaming_request(
                std::move(req),
                [loop, stream_ptr, done, completion_id, model, created,
                 is_chat, include_usage, continuous_usage, completion_tokens](
                    const domain::StreamingChunkResponse& chunk, bool is_final) {
                    if (done->load() || !*stream_ptr) {
                        return;
                    }
                    if (!chunk.choices.empty()) {
                        completion_tokens->fetch_add(1);

                        std::string sse;
                        if (is_chat) {
                            std::optional<domain::CompletionUsage> usage;
                            if (continuous_usage) {
                                int tokens = completion_tokens->load();
                                usage = domain::CompletionUsage{0, tokens, tokens};
                            }
                            sse = domain::ChatCompletionStreamChunk::makeContentChunk(
                                completion_id, model, created, chunk.choices[0], usage).toSSE();
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
                            loop->queueInLoop([stream_ptr, sse = std::move(sse)]() {
                                if (*stream_ptr) (*stream_ptr)->send(sse);
                            });
                        }
                    }
                    if (is_final) {
                        loop->queueInLoop(
                            [stream_ptr, done, is_chat, include_usage,
                             completion_id, model, created, completion_tokens]() {
                                if (!done->exchange(true) && *stream_ptr) {
                                    if (is_chat && include_usage) {
                                        int tokens = completion_tokens->load();
                                        domain::CompletionUsage usage{0, tokens, tokens};
                                        (*stream_ptr)->send(
                                            domain::ChatCompletionStreamChunk::makeUsageChunk(
                                                completion_id, model, created, usage).toSSE());
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

void LLMController::health(
    const drogon::HttpRequestPtr& /* req */,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {

    Json::Value response;
    response["status"] = "healthy";
    response["timestamp"] = static_cast<Json::Int64>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );

    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    callback(resp);
}

void LLMController::ready(
    const drogon::HttpRequestPtr& /* req */,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) const {

    auto status = service_->get_system_status();

    Json::Value response;
    response["model_ready"] = status.model_ready;
    response["queue_size"] = static_cast<Json::UInt64>(status.queue_size);
    response["max_queue_size"] = static_cast<Json::UInt64>(status.max_queue_size);
    response["device"] = status.device;

    Json::Value workers(Json::arrayValue);
    for (const auto& worker : status.worker_info) {
        Json::Value w;
        w["worker_id"] = worker.worker_id;
        w["is_ready"] = worker.is_ready;
        w["processed_requests"] = static_cast<Json::UInt64>(worker.processed_requests);
        workers.append(w);
    }
    response["workers"] = workers;

    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    if (!status.model_ready) {
        resp->setStatusCode(drogon::k503ServiceUnavailable);
    }
    callback(resp);
}

} // namespace tt::api
