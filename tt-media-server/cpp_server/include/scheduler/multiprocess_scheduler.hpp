// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <functional>
#include <future>
#include <variant>
#include <vector>
#include <sys/wait.h>
#include <signal.h>

#include "ipc/shared_memory.hpp"
#include "scheduler/thread_safe_queue.hpp"
#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"
#include "runners/base_device_runner.hpp"

namespace tt::scheduler {

/**
 * Task for multiprocess scheduler.
 * Serialized and sent to worker process via shared memory.
 */
struct ProcessTask {
    domain::CompletionRequest request;
    bool is_streaming;
    std::string task_id;
};

/**
 * Multiprocess Scheduler using shared memory ring buffers for IPC.
 *
 * Architecture:
 * - Main process: HTTP handlers, dispatches tasks, reads tokens from shm
 * - Worker processes: Each has own environment, runs device runner
 * - Shared memory ring buffers: Lock-free token streaming between processes
 *
 * Performance: ~95-98% of direct callback, but with process isolation.
 */
class MultiprocessScheduler {
public:
    static constexpr int DEFAULT_WORKER_COUNT = 4;
    static constexpr size_t RING_BUFFER_CAPACITY = 65536;

    // Environment config per worker
    struct WorkerEnvConfig {
        std::unordered_map<std::string, std::string> env_vars;
    };

    // Worker info for monitoring (matching Scheduler interface)
    struct WorkerInfo {
        std::string worker_id;
        bool is_ready;
        size_t processed_requests;
    };

    explicit MultiprocessScheduler(size_t num_workers = DEFAULT_WORKER_COUNT);
    ~MultiprocessScheduler();

    // Prevent copying
    MultiprocessScheduler(const MultiprocessScheduler&) = delete;
    MultiprocessScheduler& operator=(const MultiprocessScheduler&) = delete;

    /**
     * Start the scheduler with worker processes.
     * @param env_configs Environment configuration per worker (optional)
     */
    void start(const std::vector<WorkerEnvConfig>& env_configs = {});

    /**
     * Stop all worker processes.
     */
    void stop();

    /**
     * Check if workers are ready.
     */
    bool is_ready() const { return is_ready_.load(); }

    /**
     * Get the current queue size (pending tasks).
     */
    size_t queue_size() const { return pending_tasks_.load(); }

    /**
     * Get worker info for monitoring.
     */
    std::vector<WorkerInfo> get_worker_info() const;

    /**
     * Submit a streaming request.
     * Callback will be called from the consumer thread when tokens arrive.
     */
    void submit_streaming_request(
        domain::CompletionRequest request,
        std::function<void(const domain::StreamingChunkResponse&, bool is_final)> callback
    );

    /**
     * Submit a non-streaming request.
     */
    std::future<domain::CompletionResponse> submit_request(domain::CompletionRequest request);

    /**
     * Set the runner factory (used by worker processes).
     */
    void set_runner_factory(std::function<std::unique_ptr<runners::BaseDeviceRunner>(const std::string&)> factory) {
        runner_factory_ = std::move(factory);
    }

    /**
     * Get statistics.
     */
    struct Stats {
        uint64_t tokens_produced = 0;
        uint64_t tokens_consumed = 0;
        uint64_t tasks_submitted = 0;
        uint64_t tasks_completed = 0;
        double avg_token_latency_us = 0;
    };
    Stats get_stats() const;

private:
    /**
     * Worker process entry point.
     * Called after fork() in child process.
     */
    [[noreturn]] void worker_process_main(int worker_id, const WorkerEnvConfig& env_config);

    /**
     * Consumer thread - reads tokens from shared memory and dispatches callbacks.
     */
    void consumer_loop();

    /**
     * Dispatch a task to a worker.
     */
    void dispatch_task(ProcessTask task);

    // Worker process info
    struct WorkerProcess {
        pid_t pid = -1;
        int worker_id = -1;
        std::unique_ptr<ipc::TokenRingBuffer<RING_BUFFER_CAPACITY>> token_buffer;  // For receiving tokens
        std::unique_ptr<ipc::TokenRingBuffer<1024>> task_buffer;  // For sending tasks (smaller)
        bool is_ready = false;
    };

    std::vector<WorkerProcess> workers_;
    size_t num_workers_;

    // Consumer thread (reads from all worker buffers)
    std::thread consumer_thread_;

    // Callbacks for streaming tasks
    mutable std::mutex callbacks_mutex_;
    std::unordered_map<std::string, std::function<void(const domain::StreamingChunkResponse&, bool)>> stream_callbacks_;

    // Promises for non-streaming tasks
    mutable std::mutex promises_mutex_;
    std::unordered_map<std::string, std::shared_ptr<std::promise<domain::CompletionResponse>>> result_promises_;

    // Round-robin task distribution
    std::atomic<uint64_t> next_worker_{0};

    // Pending task counter
    std::atomic<size_t> pending_tasks_{0};

    // Runner factory (for worker processes)
    std::function<std::unique_ptr<runners::BaseDeviceRunner>(const std::string&)> runner_factory_;

    std::atomic<bool> is_ready_{false};
    std::atomic<bool> running_{false};

    // Stats
    mutable std::mutex stats_mutex_;
    Stats stats_;
};

} // namespace tt::scheduler
