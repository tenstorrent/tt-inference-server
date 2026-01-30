// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "api/llm_controller.hpp"

#include <random>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <iostream>

namespace tt::api {

LLMController::LLMController() {
    service_ = std::make_shared<services::LLMService>();
    service_->start();
    std::cout << "[LLMController] Initialized and service started" << std::endl;
}

std::string LLMController::generate_completion_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static const char* hex_chars = "0123456789abcdef";

    std::ostringstream ss;
    ss << "cmpl-";
    for (int i = 0; i < 24; ++i) {
        ss << hex_chars[dis(gen)];
    }
    return ss.str();
}

void LLMController::completions(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    // Parse JSON body
    auto json = req->getJsonObject();
    if (!json) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            Json::Value("Invalid JSON body")
        );
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    // Parse request
    domain::CompletionRequest request;
    try {
        request = domain::CompletionRequest::fromJson(*json);
        request.task_id = generate_completion_id().substr(5); // Remove "cmpl-" prefix
    } catch (const std::exception& e) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            Json::Value(std::string("Failed to parse request: ") + e.what())
        );
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    // Check if model is ready
    if (!service_->is_model_ready()) {
        Json::Value error;
        error["error"]["message"] = "Model is not ready";
        error["error"]["type"] = "service_unavailable";
        auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
        resp->setStatusCode(drogon::k503ServiceUnavailable);
        callback(resp);
        return;
    }

    // Handle streaming or non-streaming
    if (request.stream) {
        handle_streaming(request, req, std::move(callback));
    } else {
        handle_non_streaming(request, std::move(callback));
    }
}

void LLMController::handle_non_streaming(
    const domain::CompletionRequest& request,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    try {
        auto future = service_->process_request(request);

        // Wait for result (with timeout)
        auto status = future.wait_for(std::chrono::seconds(30));
        if (status == std::future_status::timeout) {
            Json::Value error;
            error["error"]["message"] = "Request timeout";
            error["error"]["type"] = "timeout";
            auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
            resp->setStatusCode(drogon::k504GatewayTimeout);
            callback(resp);
            return;
        }

        auto response = future.get();
        auto resp = drogon::HttpResponse::newHttpJsonResponse(response.toJson());
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        callback(resp);

    } catch (const std::exception& e) {
        Json::Value error;
        error["error"]["message"] = e.what();
        error["error"]["type"] = "internal_error";
        auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
        resp->setStatusCode(drogon::k500InternalServerError);
        callback(resp);
    }
}

void LLMController::handle_streaming(
    const domain::CompletionRequest& request,
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    // Capture request by value since we need it in the async callback
    auto req_copy = request;

    // Create SSE response
    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [this, req_copy](drogon::ResponseStreamPtr stream) {
            // Track completion for "data: [DONE]"
            auto done = std::make_shared<std::atomic<bool>>(false);
            // Wrap stream in shared_ptr for capturing in lambdas
            auto stream_ptr = std::make_shared<drogon::ResponseStreamPtr>(std::move(stream));

            service_->process_streaming_request(
                req_copy,
                // Chunk callback
                [stream_ptr, done](const domain::StreamingChunkResponse& chunk) {
                    if (!done->load() && *stream_ptr) {
                        std::string sse = chunk.toSSE();
                        (*stream_ptr)->send(sse);
                    }
                },
                // Done callback
                [stream_ptr, done]() {
                    if (!done->exchange(true) && *stream_ptr) {
                        (*stream_ptr)->send("data: [DONE]\n\n");
                        (*stream_ptr)->close();
                    }
                }
            );
        }
    );

    resp->setContentTypeCode(drogon::CT_TEXT_PLAIN);
    resp->addHeader("Content-Type", "text/event-stream");
    resp->addHeader("Cache-Control", "no-cache");
    resp->addHeader("Connection", "keep-alive");
    resp->addHeader("X-Accel-Buffering", "no");

    callback(resp);
}

void LLMController::health(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

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
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

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
