// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "api/llm_controller.hpp"
#include "config/settings.hpp"
#include "domain/chat_completion_request.hpp"
#include "domain/chat_completion_response.hpp"


#include <random>
#include <sstream>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

#include <json/json.h>
#include <trantor/net/EventLoop.h>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace tt::api {

LLMController::LLMController() {
    // Only initialize if TT_MODEL_SERVICE=llm or not set
    if (!tt::config::is_llm_service_enabled()) {
        std::cout << "[LLMController] Skipping initialization (TT_MODEL_SERVICE != llm)" << std::endl;
        return;
    }

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

    // Handle streaming or non-streaming (move request into streaming path to avoid copy)
    if (request.stream) {
        handle_streaming(std::move(request), req, std::move(callback));
    } else {
        handle_non_streaming(request, std::move(callback));
    }
}

Json::Value LLMController::chat_error_json(const std::string& message, const std::string& type,
    const Json::Value& param, const Json::Value& code) {
    Json::Value error;
    error["object"] = "error";
    error["message"] = message;
    error["type"] = type;
    error["param"] = param;
    error["code"] = code;
    return error;
}

void LLMController::chat_completions(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    auto json = req->getJsonObject();
    if (!json) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            chat_error_json("Invalid JSON body", "invalid_request_error"));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    domain::ChatCompletionRequest chat_req;
    try {
        chat_req = domain::ChatCompletionRequest::fromJson(*json);
        chat_req.task_id = generate_completion_id().substr(5); // Remove "cmpl-" prefix
    } catch (const std::exception& e) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            chat_error_json(std::string("Failed to parse request: ") + e.what(), "invalid_request_error"));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    if (chat_req.messages.empty()) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            chat_error_json("messages is required and must be a non-empty array", "invalid_request_error", Json::Value("messages")));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    if (!service_->is_model_ready()) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            chat_error_json("Model is not ready", "service_unavailable"));
        resp->setStatusCode(drogon::k503ServiceUnavailable);
        callback(resp);
        return;
    }

    // Convert chat request to completion request (messages -> prompt via chat template)
    domain::CompletionRequest request = chat_req.to_completion_request();
    if (request.stream) {
        handle_chat_streaming(std::move(request), req, std::move(callback));
    } else {
        handle_chat_non_streaming(request, std::move(callback));
    }
}

void LLMController::run_async_completion(
    const domain::CompletionRequest& request,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback,
    ResponseFormatter formatter) {

    auto future = service_->process_request(request);
    trantor::EventLoop* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
    if (!loop) {
        Json::Value err;
        err["error"]["message"] = "No event loop";
        err["error"]["type"] = "internal_error";
        callback(drogon::HttpResponse::newHttpJsonResponse(err));
        return;
    }

    std::thread([future = std::move(future), callback = std::move(callback), formatter = std::move(formatter), loop]() mutable {
        try {
            if (future.wait_for(std::chrono::seconds(30)) == std::future_status::timeout) {
                Json::Value err;
                err["error"]["message"] = "Request timeout";
                err["error"]["type"] = "timeout";
                auto resp = drogon::HttpResponse::newHttpJsonResponse(err);
                resp->setStatusCode(drogon::k504GatewayTimeout);
                loop->queueInLoop([callback = std::move(callback), resp]() { callback(resp); });
                return;
            }
            Json::Value json = formatter(future.get());
            auto resp = drogon::HttpResponse::newHttpJsonResponse(json);
            resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            loop->queueInLoop([callback = std::move(callback), resp]() { callback(resp); });
        } catch (...) {
            Json::Value err;
            err["error"]["type"] = "internal_error";
            try { std::rethrow_exception(std::current_exception()); } catch (const std::exception& e) { err["error"]["message"] = e.what(); } catch (...) { err["error"]["message"] = "Unknown error"; }
            auto resp = drogon::HttpResponse::newHttpJsonResponse(err);
            resp->setStatusCode(drogon::k500InternalServerError);
            loop->queueInLoop([callback = std::move(callback), resp]() { callback(resp); });
        }
    }).detach();
}

void LLMController::handle_non_streaming(
    const domain::CompletionRequest& request,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    run_async_completion(request, std::move(callback), [](const domain::CompletionResponse& r) { return r.toJson(); });
}

void LLMController::handle_chat_non_streaming(
    const domain::CompletionRequest& request,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    run_async_completion(request, std::move(callback), [](const domain::CompletionResponse& r) {
        return domain::ChatCompletionResponse::fromCompletionResponse(r).toJson();
    });
}

void LLMController::handle_chat_streaming(
    domain::CompletionRequest request,
    const drogon::HttpRequestPtr&,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    const int64_t created = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    const std::string completion_id = "chatcmpl-" + request.task_id;
    const std::string model = request.model.value_or("test-model");

    // Extract stream options before moving request
    const bool include_usage = request.stream_options.has_value()
        && request.stream_options->include_usage;
    const bool continuous_usage = request.stream_options.has_value()
        && request.stream_options->continuous_usage_stats;

    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [this, req = std::move(request), completion_id, model, created,
         include_usage, continuous_usage](
            drogon::ResponseStreamPtr stream) mutable {
            trantor::EventLoop* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
            auto done = std::make_shared<std::atomic<bool>>(false);
            auto stream_ptr =
                std::make_shared<drogon::ResponseStreamPtr>(std::move(stream));
            auto completion_tokens = std::make_shared<std::atomic<int>>(0);

            // Send initial role-only chunk before content generation starts
            {
                std::optional<domain::CompletionUsage> initial_usage;
                if (continuous_usage) {
                    initial_usage = domain::CompletionUsage{0, 0, 0};
                }
                auto initial_chunk = domain::ChatCompletionStreamChunk::makeInitialChunk(
                    completion_id, model, created, initial_usage);
                if (*stream_ptr) (*stream_ptr)->send(initial_chunk.toSSE());
            }

            service_->process_streaming_request(
                std::move(req),
                [loop, stream_ptr, done, completion_id, model, created,
                 completion_tokens, continuous_usage](
                    const domain::StreamingChunkResponse& chunk) {
                    if (!done->load() && *stream_ptr && !chunk.choices.empty()) {
                        completion_tokens->fetch_add(1);

                        std::optional<domain::CompletionUsage> usage;
                        if (continuous_usage) {
                            int tokens = completion_tokens->load();
                            usage = domain::CompletionUsage{0, tokens, tokens};
                        }

                        auto chat_chunk = domain::ChatCompletionStreamChunk::makeContentChunk(
                            completion_id, model, created, chunk.choices[0], usage);

                        std::string sse = chat_chunk.toSSE();
                        loop->queueInLoop([stream_ptr, sse]() {
                            if (*stream_ptr) (*stream_ptr)->send(sse);
                        });
                    }
                },
                [loop, stream_ptr, done, include_usage,
                 completion_id, model, created, completion_tokens]() {
                    loop->queueInLoop(
                        [stream_ptr, done, include_usage,
                         completion_id, model, created, completion_tokens]() {
                        if (!done->exchange(true) && *stream_ptr) {
                            if (include_usage) {
                                int tokens = completion_tokens->load();
                                domain::CompletionUsage usage{0, tokens, tokens};
                                auto usage_chunk = domain::ChatCompletionStreamChunk::makeUsageChunk(
                                    completion_id, model, created, usage);
                                (*stream_ptr)->send(usage_chunk.toSSE());
                            }

                            (*stream_ptr)->send("data: [DONE]\n\n");
                            (*stream_ptr)->close();
                        }
                    });
                });
        });

    resp->setContentTypeString("text/event-stream");
    resp->addHeader("Cache-Control", "no-cache");
    resp->addHeader("Connection", "keep-alive");
    resp->addHeader("X-Accel-Buffering", "no");

    callback(resp);
}

void LLMController::handle_streaming(
    domain::CompletionRequest request,
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    const int64_t created = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    const std::string completion_id = "cmpl-" + request.task_id;
    const std::string model = request.model.value_or("test-model");
    const std::string task_id = request.task_id;

    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [this, req = std::move(request), completion_id, model, created, task_id](
            drogon::ResponseStreamPtr stream) mutable {
            auto done = std::make_shared<std::atomic<bool>>(false);
            auto stream_ptr =
                std::make_shared<drogon::ResponseStreamPtr>(std::move(stream));

            // Reusable RapidJSON envelope: build once, update only choices[0] per chunk (shared for copyable lambda)
            auto envelope = std::make_shared<rapidjson::Document>(rapidjson::kObjectType);
            rapidjson::Document::AllocatorType& a = envelope->GetAllocator();
            envelope->AddMember("id", rapidjson::Value(completion_id.c_str(), a).Move(), a);
            envelope->AddMember("object", "text_completion", a);
            envelope->AddMember("created", static_cast<int64_t>(created), a);
            envelope->AddMember("model", rapidjson::Value(model.c_str(), a).Move(), a);
            rapidjson::Value choices(rapidjson::kArrayType);
            rapidjson::Value choice(rapidjson::kObjectType);
            choice.AddMember("text", rapidjson::Value("", a).Move(), a);
            choice.AddMember("index", 0, a);
            choice.AddMember("finish_reason", rapidjson::Value(rapidjson::kNullType), a);
            choices.PushBack(choice.Move(), a);
            envelope->AddMember("choices", choices.Move(), a);

            service_->process_streaming_request(
                std::move(req),
                // Chunk callback - serialize with RapidJSON and send immediately
                [stream_ptr, done, envelope](
                    const domain::StreamingChunkResponse& chunk) {
                    if (!done->load() && *stream_ptr && !chunk.choices.empty()) {
                        // Skip sending final empty chunk as SSE data (client counts content tokens only)
                        if (chunk.choices[0].text.empty() && chunk.choices[0].finish_reason.has_value()) {
                            return;
                        }
                        rapidjson::Value& choice = (*envelope)["choices"][0];
                        rapidjson::Document::AllocatorType& alloc = envelope->GetAllocator();
                        choice["text"].SetString(
                            chunk.choices[0].text.c_str(),
                            static_cast<rapidjson::SizeType>(chunk.choices[0].text.size()),
                            alloc);
                        choice["index"].SetInt(chunk.choices[0].index);
                        if (chunk.choices[0].finish_reason.has_value()) {
                            choice["finish_reason"].SetString(
                                chunk.choices[0].finish_reason->c_str(),
                                static_cast<rapidjson::SizeType>(chunk.choices[0].finish_reason->size()),
                                alloc);
                        } else {
                            choice["finish_reason"].SetNull();
                        }

                        static thread_local rapidjson::StringBuffer buf;
                        buf.Clear();
                        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
                        envelope->Accept(writer);
                        std::string sse = "data: ";
                        sse.append(buf.GetString(), buf.GetSize());
                        sse.append("\n\n");

                        (*stream_ptr)->send(sse);
                    }
                },
                // Done callback - send termination and close
                [stream_ptr, done]() {
                    if (!done->exchange(true) && *stream_ptr) {
                        (*stream_ptr)->send("data: [DONE]\n\n");
                        (*stream_ptr)->close();
                    }
                });
        });

    resp->setContentTypeString("text/event-stream");
    resp->addHeader("Cache-Control", "no-cache");
    resp->addHeader("Connection", "keep-alive");
    resp->addHeader("X-Accel-Buffering", "no");

    callback(resp);
}

void LLMController::handle_streaming_buffered(
    domain::CompletionRequest request,
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    const int64_t created = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    const std::string completion_id = "cmpl-" + request.task_id;
    const std::string model = request.model.value_or("test-model");

    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [this, req = std::move(request), completion_id, model, created](
            drogon::ResponseStreamPtr stream) mutable {
            auto done = std::make_shared<std::atomic<bool>>(false);
            auto stream_ptr =
                std::make_shared<drogon::ResponseStreamPtr>(std::move(stream));

            auto envelope = std::make_shared<rapidjson::Document>(rapidjson::kObjectType);
            rapidjson::Document::AllocatorType& a = envelope->GetAllocator();
            envelope->AddMember("id", rapidjson::Value(completion_id.c_str(), a).Move(), a);
            envelope->AddMember("object", "text_completion", a);
            envelope->AddMember("created", static_cast<int64_t>(created), a);
            envelope->AddMember("model", rapidjson::Value(model.c_str(), a).Move(), a);
            rapidjson::Value choices(rapidjson::kArrayType);
            rapidjson::Value choice(rapidjson::kObjectType);
            choice.AddMember("text", rapidjson::Value("", a).Move(), a);
            choice.AddMember("index", 0, a);
            choice.AddMember("finish_reason", rapidjson::Value(rapidjson::kNullType), a);
            choices.PushBack(choice.Move(), a);
            envelope->AddMember("choices", choices.Move(), a);

            // 32KB write buffer to reduce syscalls for high-throughput scenarios
            auto buffer = std::make_shared<std::string>();
            buffer->reserve(64 * 1024);
            auto buffer_mutex = std::make_shared<std::mutex>();
            static constexpr size_t FLUSH_THRESHOLD = 32 * 1024;

            service_->process_streaming_request(
                std::move(req),
                [stream_ptr, done, envelope, buffer, buffer_mutex](                    const domain::StreamingChunkResponse& chunk) {
                    if (!done->load() && *stream_ptr && !chunk.choices.empty()) {
                        // Skip sending final empty chunk as SSE data (client counts content tokens only)
                        if (chunk.choices[0].text.empty() && chunk.choices[0].finish_reason.has_value()) {
                            return;
                        }
                        rapidjson::Value& choice = (*envelope)["choices"][0];
                        rapidjson::Document::AllocatorType& alloc = envelope->GetAllocator();
                        choice["text"].SetString(
                            chunk.choices[0].text.c_str(),
                            static_cast<rapidjson::SizeType>(chunk.choices[0].text.size()),
                            alloc);
                        choice["index"].SetInt(chunk.choices[0].index);
                        if (chunk.choices[0].finish_reason.has_value()) {
                            choice["finish_reason"].SetString(
                                chunk.choices[0].finish_reason->c_str(),
                                static_cast<rapidjson::SizeType>(chunk.choices[0].finish_reason->size()),
                                alloc);
                        } else {
                            choice["finish_reason"].SetNull();
                        }

                        static thread_local rapidjson::StringBuffer buf;
                        buf.Clear();
                        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
                        envelope->Accept(writer);
                        std::string sse = "data: ";
                        sse.append(buf.GetString(), buf.GetSize());
                        sse.append("\n\n");

                        std::lock_guard<std::mutex> lock(*buffer_mutex);
                        buffer->append(sse);

                        if (buffer->size() >= FLUSH_THRESHOLD) {
                            (*stream_ptr)->send(*buffer);
                            buffer->clear();
                        }
                    }
                },
                [stream_ptr, done, buffer, buffer_mutex]() {
                    if (!done->exchange(true) && *stream_ptr) {
                        std::lock_guard<std::mutex> lock(*buffer_mutex);
                        if (!buffer->empty()) {
                            (*stream_ptr)->send(*buffer);
                            buffer->clear();
                        }
                        (*stream_ptr)->send("data: [DONE]\n\n");
                        (*stream_ptr)->close();
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
