// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include "services/base_service.hpp"
#include "ipc/shared_memory.hpp"
#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"
#include "runners/llm_engine/engine/boost_ipc_task_queue.hpp"

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tt::services {

/**
 * LLM Service for text completions.
 *
 * Owns multiprocess worker management end-to-end:
 *   - Forks worker processes (each with its own TT_VISIBLE_DEVICES)
 *   - Dispatches tasks via Boost.Interprocess message queue
 *   - Streams tokens back through shared-memory ring buffers
 *   - Runs per-worker consumer threads that translate tokens into callbacks
 *
 * Replaces the former BaseService + MultiprocessScheduler split.
 */
class LLMService : public BaseService {
public:
    static constexpr size_t RING_BUFFER_CAPACITY = 65536;

    /** Per-worker environment configuration. */
    struct WorkerEnvConfig {
        std::unordered_map<std::string, std::string> env_vars;
    };

    /** Scheduler-level statistics. */
    struct Stats {
        uint64_t tokens_produced = 0;
        uint64_t tokens_consumed = 0;
        uint64_t tasks_submitted = 0;
        uint64_t tasks_completed = 0;
        double avg_token_latency_us = 0;
    };

    LLMService();
    ~LLMService() override;

    // Non-copyable
    LLMService(const LLMService&) = delete;
    LLMService& operator=(const LLMService&) = delete;

    // --- BaseService interface ---
    void start() override;
    void stop() override;
    bool is_model_ready() const override;
    SystemStatus get_system_status() const override;

    std::future<domain::CompletionResponse> process_request(domain::CompletionRequest request) override;
    void process_streaming_request(
        domain::CompletionRequest request,
        std::function<void(const domain::StreamingChunkResponse&)> chunk_callback,
        std::function<void()> done_callback
    ) override;

    /** Get scheduler statistics. */
    Stats get_stats() const;

private:
    /** Internal task dispatched to a worker. */
    struct ProcessTask {
        domain::CompletionRequest request;
        bool is_streaming;
        std::string task_id;
    };

    /** State for a single forked worker process.
     *  is_ready / is_alive are written once (start or crash detection)
     *  and read for monitoring — plain bool is sufficient. */
    struct WorkerProcess {
        pid_t pid = -1;
        int worker_id = -1;
        std::unique_ptr<ipc::TokenRingBuffer<RING_BUFFER_CAPACITY>> token_buffer;
        std::shared_ptr<llm_engine::BoostIpcTaskQueue> task_buffer;
        bool is_ready = false;
        bool is_alive = true;
    };

    /**
     * Worker process entry point (runs after fork in child process).
     */
    [[noreturn]] void worker_process_main(int worker_id, const WorkerEnvConfig& env_config);

    /**
     * Consumer thread body — reads tokens from one worker's ring buffer
     * and dispatches streaming / non-streaming callbacks.
     */
    void consumer_loop_for_worker(size_t worker_idx);

    /**
     * Check if a worker process has exited (non-blocking waitpid).
     * If crashed, logs the event, marks the worker dead, and fails all
     * pending callbacks/promises that were routed to it.
     * @return true if the worker is still alive.
     */
    bool check_worker_alive(size_t worker_idx);

    /**
     * Fail all pending streaming callbacks and non-streaming promises
     * whose tasks were dispatched (best-effort: fails every pending entry
     * since tasks are not per-worker tracked).
     */
    void fail_all_pending(const std::string& reason);

    /**
     * Round-robin dispatch a task to a worker.
     */
    void dispatch_task(ProcessTask task);

    /**
     * Internal: submit a streaming request (generates task ID, registers callback, dispatches).
     */
    void submit_streaming_request(
        domain::CompletionRequest request,
        std::function<void(const domain::StreamingChunkResponse&, bool is_final)> callback
    );

    /**
     * Internal: submit a non-streaming request.
     */
    std::future<domain::CompletionResponse> submit_request(domain::CompletionRequest request);

    // Worker processes
    std::vector<WorkerProcess> workers_;
    size_t num_workers_;

    // One consumer thread per worker for parallel token processing
    std::vector<std::thread> consumer_threads_;

    // Streaming callbacks keyed by task_id
    mutable std::mutex callbacks_mutex_;
    std::unordered_map<std::string,
        std::function<void(const domain::StreamingChunkResponse&, bool)>> stream_callbacks_;

    // Promises for non-streaming completions keyed by task_id
    mutable std::mutex promises_mutex_;
    std::unordered_map<std::string,
        std::shared_ptr<std::promise<domain::CompletionResponse>>> result_promises_;

    // Round-robin worker selection
    std::atomic<uint64_t> next_worker_{0};

    // Pending task counter (exposed as queue_size)
    std::atomic<size_t> pending_tasks_{0};

    std::atomic<bool> is_ready_{false};
    std::atomic<bool> running_{false};

    // Stats
    mutable std::mutex stats_mutex_;
    Stats stats_;

    // Service-level configuration
    size_t max_queue_size_ = 10000;
    std::string device_ = "cpu";
};

} // namespace tt::services
