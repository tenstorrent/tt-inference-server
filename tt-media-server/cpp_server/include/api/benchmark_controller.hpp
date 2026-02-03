// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <drogon/HttpController.h>
#include <memory>

#include "scheduler/scheduler.hpp"
#include "scheduler/multiprocess_scheduler.hpp"
#include "runners/llm_test_runner.hpp"

namespace tt::api {

/**
 * Benchmark controller for comparing threading vs multiprocessing performance.
 *
 * Endpoints:
 *   POST /benchmark/thread/completions  - Use threaded scheduler
 *   POST /benchmark/mp/completions      - Use multiprocess scheduler
 *   GET  /benchmark/stats               - Get performance stats
 */
class BenchmarkController : public drogon::HttpController<BenchmarkController> {
public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(BenchmarkController::thread_completions, "/benchmark/thread/completions", drogon::Post);
    ADD_METHOD_TO(BenchmarkController::mp_completions, "/benchmark/mp/completions", drogon::Post);
    ADD_METHOD_TO(BenchmarkController::get_stats, "/benchmark/stats", drogon::Get);
    ADD_METHOD_TO(BenchmarkController::init_schedulers, "/benchmark/init", drogon::Post);
    METHOD_LIST_END

    BenchmarkController();
    ~BenchmarkController();

    void thread_completions(
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    void mp_completions(
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    void get_stats(
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    void init_schedulers(
        const drogon::HttpRequestPtr& req,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

private:
    void handle_streaming_thread(
        const domain::CompletionRequest& request,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    void handle_streaming_mp(
        const domain::CompletionRequest& request,
        std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    std::string generate_task_id();

    // Threaded scheduler (direct callbacks)
    std::unique_ptr<scheduler::Scheduler> thread_scheduler_;

    // Multiprocess scheduler (shared memory IPC)
    std::unique_ptr<scheduler::MultiprocessScheduler> mp_scheduler_;

    // Stats tracking
    struct BenchmarkStats {
        std::atomic<uint64_t> thread_requests{0};
        std::atomic<uint64_t> thread_tokens{0};
        std::atomic<uint64_t> thread_time_us{0};

        std::atomic<uint64_t> mp_requests{0};
        std::atomic<uint64_t> mp_tokens{0};
        std::atomic<uint64_t> mp_time_us{0};
    };
    BenchmarkStats stats_;

    bool initialized_ = false;
};

} // namespace tt::api
