// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "api/benchmark_controller.hpp"

#include <random>
#include <sstream>
#include <chrono>
#include <iostream>

namespace tt::api {

BenchmarkController::BenchmarkController() {
    std::cout << "[BenchmarkController] Created - call POST /benchmark/init to start schedulers\n";
}

BenchmarkController::~BenchmarkController() {
    if (thread_scheduler_) {
        thread_scheduler_->stop();
    }
    if (mp_scheduler_) {
        mp_scheduler_->stop();
    }
}

std::string BenchmarkController::generate_task_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static const char* hex_chars = "0123456789abcdef";
    static std::atomic<uint64_t> counter{0};

    std::ostringstream ss;
    ss << "bench-" << counter.fetch_add(1) << "-";
    for (int i = 0; i < 8; ++i) {
        ss << hex_chars[dis(gen)];
    }
    return ss.str();
}

void BenchmarkController::init_schedulers(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    if (initialized_) {
        Json::Value response;
        response["status"] = "already_initialized";
        auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
        callback(resp);
        return;
    }

    auto json = req->getJsonObject();
    int worker_count = 4;  // Default
    if (json && json->isMember("worker_count")) {
        worker_count = (*json)["worker_count"].asInt();
    }

    std::cout << "[BenchmarkController] Initializing schedulers with " << worker_count << " workers\n";

    // Runner factory for both schedulers
    auto runner_factory = [](const std::string& device_id) {
        return std::make_unique<runners::LLMTestRunner>(device_id);
    };

    // Initialize threaded scheduler
    thread_scheduler_ = std::make_unique<scheduler::Scheduler>();
    thread_scheduler_->set_runner_factory(runner_factory);
    thread_scheduler_->start();

    // Initialize multiprocess scheduler with per-worker environment
    mp_scheduler_ = std::make_unique<scheduler::MultiprocessScheduler>(worker_count);
    mp_scheduler_->set_runner_factory(runner_factory);

    // Configure different environment for each worker
    std::vector<scheduler::MultiprocessScheduler::WorkerEnvConfig> env_configs;
    for (int i = 0; i < worker_count; i++) {
        scheduler::MultiprocessScheduler::WorkerEnvConfig config;
        config.env_vars["TT_DEVICE_ID"] = std::to_string(i);
        config.env_vars["TT_DEVICE_MESH"] = "[" + std::to_string(i) + "]";
        config.env_vars["WORKER_TYPE"] = "benchmark";
        env_configs.push_back(config);
    }

    mp_scheduler_->start(env_configs);

    initialized_ = true;

    Json::Value response;
    response["status"] = "initialized";
    response["thread_workers"] = worker_count;
    response["mp_workers"] = worker_count;
    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    callback(resp);
}

void BenchmarkController::thread_completions(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    if (!initialized_ || !thread_scheduler_) {
        Json::Value error;
        error["error"] = "Scheduler not initialized. Call POST /benchmark/init first.";
        auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
        resp->setStatusCode(drogon::k503ServiceUnavailable);
        callback(resp);
        return;
    }

    auto json = req->getJsonObject();
    if (!json) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(Json::Value("Invalid JSON"));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    domain::CompletionRequest request;
    try {
        request = domain::CompletionRequest::fromJson(*json);
        request.task_id = generate_task_id();
    } catch (const std::exception& e) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(Json::Value(e.what()));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    if (request.stream) {
        handle_streaming_thread(request, std::move(callback));
    } else {
        // Non-streaming
        auto future = thread_scheduler_->submit_request(request);
        auto response = future.get();
        auto resp = drogon::HttpResponse::newHttpJsonResponse(response.toJson());
        callback(resp);
    }
}

void BenchmarkController::mp_completions(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    if (!initialized_ || !mp_scheduler_) {
        Json::Value error;
        error["error"] = "Scheduler not initialized. Call POST /benchmark/init first.";
        auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
        resp->setStatusCode(drogon::k503ServiceUnavailable);
        callback(resp);
        return;
    }

    auto json = req->getJsonObject();
    if (!json) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(Json::Value("Invalid JSON"));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    domain::CompletionRequest request;
    try {
        request = domain::CompletionRequest::fromJson(*json);
        request.task_id = generate_task_id();
    } catch (const std::exception& e) {
        auto resp = drogon::HttpResponse::newHttpJsonResponse(Json::Value(e.what()));
        resp->setStatusCode(drogon::k400BadRequest);
        callback(resp);
        return;
    }

    if (request.stream) {
        handle_streaming_mp(request, std::move(callback));
    } else {
        auto future = mp_scheduler_->submit_request(request);
        auto response = future.get();
        auto resp = drogon::HttpResponse::newHttpJsonResponse(response.toJson());
        callback(resp);
    }
}

void BenchmarkController::handle_streaming_thread(
    const domain::CompletionRequest& request,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    auto req_copy = request;
    auto start_time = std::make_shared<std::chrono::high_resolution_clock::time_point>(
        std::chrono::high_resolution_clock::now()
    );
    auto token_count = std::make_shared<std::atomic<int>>(0);
    auto stats_ptr = &stats_;

    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [this, req_copy, start_time, token_count, stats_ptr](drogon::ResponseStreamPtr stream) {
            auto done = std::make_shared<std::atomic<bool>>(false);
            auto stream_ptr = std::make_shared<drogon::ResponseStreamPtr>(std::move(stream));

            // Buffer for batching
            auto buffer = std::make_shared<std::string>();
            buffer->reserve(64 * 1024);
            auto buffer_mutex = std::make_shared<std::mutex>();
            static constexpr size_t FLUSH_THRESHOLD = 32 * 1024;

            thread_scheduler_->submit_streaming_request(
                req_copy,
                [stream_ptr, done, buffer, buffer_mutex, token_count](
                    const domain::StreamingChunkResponse& chunk, bool is_final) {

                    if (done->load()) return;

                    token_count->fetch_add(1);

                    if (!is_final) {
                        std::string sse = chunk.toSSE();
                        std::lock_guard<std::mutex> lock(*buffer_mutex);
                        buffer->append(sse);

                        if (buffer->size() >= FLUSH_THRESHOLD) {
                            (*stream_ptr)->send(*buffer);
                            buffer->clear();
                        }
                    } else {
                        done->store(true);
                        std::lock_guard<std::mutex> lock(*buffer_mutex);
                        if (!buffer->empty()) {
                            (*stream_ptr)->send(*buffer);
                            buffer->clear();
                        }
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

void BenchmarkController::handle_streaming_mp(
    const domain::CompletionRequest& request,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    auto req_copy = request;
    auto start_time = std::make_shared<std::chrono::high_resolution_clock::time_point>(
        std::chrono::high_resolution_clock::now()
    );
    auto token_count = std::make_shared<std::atomic<int>>(0);
    auto stats_ptr = &stats_;

    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [this, req_copy, start_time, token_count, stats_ptr](drogon::ResponseStreamPtr stream) {
            auto done = std::make_shared<std::atomic<bool>>(false);
            auto stream_ptr = std::make_shared<drogon::ResponseStreamPtr>(std::move(stream));

            // Buffer for batching
            auto buffer = std::make_shared<std::string>();
            buffer->reserve(64 * 1024);
            auto buffer_mutex = std::make_shared<std::mutex>();
            static constexpr size_t FLUSH_THRESHOLD = 32 * 1024;

            mp_scheduler_->submit_streaming_request(
                req_copy,
                [stream_ptr, done, buffer, buffer_mutex, token_count](
                    const domain::StreamingChunkResponse& chunk, bool is_final) {

                    if (done->load()) return;

                    token_count->fetch_add(1);

                    if (!is_final) {
                        std::string sse = chunk.toSSE();
                        std::lock_guard<std::mutex> lock(*buffer_mutex);
                        buffer->append(sse);

                        if (buffer->size() >= FLUSH_THRESHOLD) {
                            (*stream_ptr)->send(*buffer);
                            buffer->clear();
                        }
                    } else {
                        done->store(true);
                        std::lock_guard<std::mutex> lock(*buffer_mutex);
                        if (!buffer->empty()) {
                            (*stream_ptr)->send(*buffer);
                            buffer->clear();
                        }
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

void BenchmarkController::get_stats(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {

    Json::Value response;

    response["initialized"] = initialized_;

    if (thread_scheduler_) {
        response["thread"]["ready"] = thread_scheduler_->is_ready();
        response["thread"]["queue_size"] = static_cast<Json::UInt64>(thread_scheduler_->queue_size());
    }

    if (mp_scheduler_) {
        response["multiprocess"]["ready"] = mp_scheduler_->is_ready();
        auto mp_stats = mp_scheduler_->get_stats();
        response["multiprocess"]["tokens_produced"] = static_cast<Json::UInt64>(mp_stats.tokens_produced);
        response["multiprocess"]["tokens_consumed"] = static_cast<Json::UInt64>(mp_stats.tokens_consumed);
        response["multiprocess"]["tasks_submitted"] = static_cast<Json::UInt64>(mp_stats.tasks_submitted);
        response["multiprocess"]["tasks_completed"] = static_cast<Json::UInt64>(mp_stats.tasks_completed);
    }

    auto resp = drogon::HttpResponse::newHttpJsonResponse(response);
    callback(resp);
}

} // namespace tt::api
