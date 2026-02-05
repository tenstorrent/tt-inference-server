// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "api/embedding_controller.hpp"
#include "config/settings.hpp"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <random>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace tt::api {

namespace {
    // Generate random hex string for task IDs
    std::string random_hex(size_t length) {
        static const char hex_chars[] = "0123456789abcdef";
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dist(0, 15);

        std::string result;
        result.reserve(length);
        for (size_t i = 0; i < length; ++i) {
            result += hex_chars[dist(gen)];
        }
        return result;
    }

    // Simple thread pool for handling response callbacks
    class CallbackThreadPool {
    public:
        CallbackThreadPool(size_t num_threads = 8) : stop_(false) {
            for (size_t i = 0; i < num_threads; ++i) {
                workers_.emplace_back([this] {
                    while (true) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(mutex_);
                            cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                            if (stop_ && tasks_.empty()) return;
                            task = std::move(tasks_.front());
                            tasks_.pop();
                        }
                        task();
                    }
                });
            }
        }

        ~CallbackThreadPool() {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
            }
            cv_.notify_all();
            for (auto& worker : workers_) {
                if (worker.joinable()) worker.join();
            }
        }

        void submit(std::function<void()> task) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                tasks_.push(std::move(task));
            }
            cv_.notify_one();
        }

    private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        std::mutex mutex_;
        std::condition_variable cv_;
        bool stop_;
    };

    // Global thread pool for callbacks
    CallbackThreadPool& get_callback_pool() {
        static CallbackThreadPool pool(16);  // 16 threads for handling callbacks
        return pool;
    }
}

EmbeddingController::EmbeddingController() {
    // Only initialize if TT_MODEL_SERVICE=embedding
    if (!tt::config::is_embedding_service()()) {
        return;
    }

    std::cout << "[EmbeddingController] Creating service...\n";

    service_ = std::make_shared<services::EmbeddingService>();
    service_->start();

    std::cout << "[EmbeddingController] Initialized and service started\n";
}

EmbeddingController::~EmbeddingController() {
    if (service_) {
        service_->stop();
    }
}

std::string EmbeddingController::generate_task_id() {
    return random_hex(24);
}

void EmbeddingController::create_embedding(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    auto start_time = std::chrono::steady_clock::now();

    // Parse request body
    auto json = req->getJsonObject();
    if (!json) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(
            Json::Value("Invalid JSON body"));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    // Validate required fields
    if (!json->isMember("input")) {
        Json::Value error;
        error["error"]["message"] = "Missing required field: input";
        error["error"]["type"] = "invalid_request_error";
        auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    // Build request
    domain::EmbeddingRequest request = domain::EmbeddingRequest::from_json(*json);
    request.task_id = generate_task_id();

    // Default model if not specified
    if (request.model.empty()) {
        request.model = "BAAI/bge-large-en-v1.5";
    }

    uint64_t req_num = request_counter_.fetch_add(1);

    auto submit_time = std::chrono::steady_clock::now();

    // Submit request and get future
    auto future = std::make_shared<std::future<domain::EmbeddingResponse>>(
        service_->process_request(std::move(request)));

    // Use thread pool instead of creating new thread per request
    get_callback_pool().submit([callback = std::move(callback), future, req_num, start_time, submit_time]() {
        try {
            auto response = future->get();
            auto got_response_time = std::chrono::steady_clock::now();

            if (!response.error.empty()) {
                Json::Value error;
                error["error"]["message"] = response.error;
                error["error"]["type"] = "server_error";
                auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
                resp->setStatusCode(drogon::k500InternalServerError);
                callback(resp);
                return;
            }

            // Build OpenAI-compatible response
            Json::Value json_response = response.to_openai_json();
            auto built_json_time = std::chrono::steady_clock::now();

            auto resp = drogon::HttpResponse::newHttpJsonResponse(json_response);

            // Log timing every 100 requests
            if (req_num % 100 == 0) {
                double parse_ms = std::chrono::duration<double, std::milli>(submit_time - start_time).count();
                double wait_ms = std::chrono::duration<double, std::milli>(got_response_time - submit_time).count();
                double build_ms = std::chrono::duration<double, std::milli>(built_json_time - got_response_time).count();
                double total_ms = std::chrono::duration<double, std::milli>(built_json_time - start_time).count();
                std::cout << "[CTRL] req=" << req_num
                          << " parse=" << parse_ms << "ms"
                          << " wait=" << wait_ms << "ms"
                          << " build=" << build_ms << "ms"
                          << " total=" << total_ms << "ms\n";
            }

            callback(resp);

        } catch (const std::exception& e) {
            Json::Value error;
            error["error"]["message"] = std::string("Internal error: ") + e.what();
            error["error"]["type"] = "server_error";
            auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
            resp->setStatusCode(drogon::k500InternalServerError);
            callback(resp);
        }
    });
}

void EmbeddingController::health(
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

void EmbeddingController::ready(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    auto status = service_->get_system_status();

    Json::Value response;
    response["model_ready"] = status.model_ready;
    response["queue_size"] = static_cast<Json::UInt64>(status.queue_size);
    response["max_queue_size"] = static_cast<Json::UInt64>(status.max_queue_size);
    response["device"] = status.device;
    response["num_workers"] = static_cast<Json::UInt64>(status.num_workers);

    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);

    if (!status.model_ready) {
        resp->setStatusCode(drogon::k503ServiceUnavailable);
    }

    callback(resp);
}

} // namespace tt::api
