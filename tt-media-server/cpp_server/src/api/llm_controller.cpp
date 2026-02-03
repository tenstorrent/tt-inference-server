// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "api/llm_controller.hpp"

#include <random>
#include <sstream>
#include <iomanip>
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

void LLMController::handle_non_streaming(
    const domain::CompletionRequest& request,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    auto future = service_->process_request(request);
    trantor::EventLoop* loop = trantor::EventLoop::getEventLoopOfCurrentThread();
    if (!loop) {
        Json::Value error;
        error["error"]["message"] = "No event loop";
        error["error"]["type"] = "internal_error";
        callback(drogon::HttpResponse::newHttpJsonResponse(error));
        return;
    }

    std::thread([future = std::move(future),
                 callback = std::move(callback),
                 loop]() mutable {
        try {
            auto status = future.wait_for(std::chrono::seconds(30));
            if (status == std::future_status::timeout) {
                Json::Value error;
                error["error"]["message"] = "Request timeout";
                error["error"]["type"] = "timeout";
                auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
                resp->setStatusCode(drogon::k504GatewayTimeout);
                loop->queueInLoop([callback = std::move(callback), resp]() {
                    callback(resp);
                });
                return;
            }

            domain::CompletionResponse response = future.get();
            auto resp = drogon::HttpResponse::newHttpJsonResponse(response.toJson());
            resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            loop->queueInLoop([callback = std::move(callback), resp]() {
                callback(resp);
            });
        } catch (...) {
            Json::Value error;
            try {
                std::rethrow_exception(std::current_exception());
            } catch (const std::exception& e) {
                error["error"]["message"] = e.what();
            } catch (...) {
                error["error"]["message"] = "Unknown error";
            }
            error["error"]["type"] = "internal_error";
            auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
            resp->setStatusCode(drogon::k500InternalServerError);
            loop->queueInLoop([callback = std::move(callback), resp]() {
                callback(resp);
            });
        }
    }).detach();
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

            // Buffer for batching tokens - reduces syscalls dramatically
            auto buffer = std::make_shared<std::string>();
            buffer->reserve(64 * 1024);  // 64KB buffer
            auto buffer_mutex = std::make_shared<std::mutex>();
            static constexpr size_t FLUSH_THRESHOLD = 32 * 1024;  // Flush at 32KB

            // Timing stats
            auto flush_count = std::make_shared<std::atomic<int>>(0);
            auto total_flush_time_us = std::make_shared<std::atomic<long long>>(0);
            auto total_bytes_sent = std::make_shared<std::atomic<long long>>(0);

            service_->process_streaming_request(
                std::move(req),
                // Chunk callback - serialize with RapidJSON, buffer and flush when large enough
                [stream_ptr, done, envelope, buffer, buffer_mutex, flush_count, total_flush_time_us, total_bytes_sent](
                    const domain::StreamingChunkResponse& chunk) {
                    if (!done->load() && *stream_ptr && !chunk.choices.empty()) {
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
                            auto start = std::chrono::high_resolution_clock::now();
                            size_t bytes = buffer->size();
                            (*stream_ptr)->send(*buffer);
                            buffer->clear();
                            auto end = std::chrono::high_resolution_clock::now();
                            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                            flush_count->fetch_add(1);
                            total_flush_time_us->fetch_add(duration_us);
                            total_bytes_sent->fetch_add(bytes);
                        }
                    }
                },
                // Done callback - flush remaining and close
                [stream_ptr, done, buffer, buffer_mutex, flush_count, total_flush_time_us, total_bytes_sent, task_id]() {
                    if (!done->exchange(true) && *stream_ptr) {
                        std::lock_guard<std::mutex> lock(*buffer_mutex);
                        if (!buffer->empty()) {
                            auto start = std::chrono::high_resolution_clock::now();
                            size_t bytes = buffer->size();
                            (*stream_ptr)->send(*buffer);
                            buffer->clear();
                            auto end = std::chrono::high_resolution_clock::now();
                            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                            flush_count->fetch_add(1);
                            total_flush_time_us->fetch_add(duration_us);
                            total_bytes_sent->fetch_add(bytes);
                        }
                        (*stream_ptr)->send("data: [DONE]\n\n");
                        (*stream_ptr)->close();

                        int flushes = flush_count->load();
                        long long total_us = total_flush_time_us->load();
                        long long bytes = total_bytes_sent->load();
                        std::cout << "[FLUSH] task=" << task_id
                                  << " flushes=" << flushes
                                  << " total_time=" << total_us << "µs"
                                  << " bytes=" << bytes
                                  << " avg_flush=" << (flushes > 0 ? total_us / flushes : 0) << "µs"
                                  << std::endl;
                    }
                });
        });

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
