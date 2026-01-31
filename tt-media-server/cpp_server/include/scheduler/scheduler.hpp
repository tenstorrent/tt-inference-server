// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <functional>
#include <future>

#include "scheduler/thread_safe_queue.hpp"
#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"
#include "runners/base_device_runner.hpp"

namespace tt::scheduler {

/**
 * Task wrapper containing request and result promise.
 */
struct SchedulerTask {
    domain::CompletionRequest request;
    bool is_streaming;

    // For non-streaming: single result promise
    std::shared_ptr<std::promise<domain::CompletionResponse>> result_promise;

    // For streaming: callback for each chunk
    std::function<void(const domain::StreamingChunkResponse&, bool is_final)> stream_callback;
};

/**
 * Scheduler manages device workers and request queuing.
 * Similar to the Python Scheduler class.
 */
class Scheduler {
public:
    static constexpr int DEFAULT_WORKER_COUNT = 1;
    static constexpr size_t DEFAULT_QUEUE_SIZE = 10000;

    Scheduler();
    ~Scheduler();

    // Prevent copying
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    /**
     * Start the scheduler and worker threads.
     */
    void start();

    /**
     * Stop the scheduler and all workers.
     */
    void stop();

    /**
     * Check if the model/workers are ready.
     */
    bool is_ready() const { return is_ready_.load(); }

    /**
     * Get the current queue size.
     */
    size_t queue_size() const { return task_queue_.size(); }

    /**
     * Submit a non-streaming request and get a future for the result.
     */
    std::future<domain::CompletionResponse> submit_request(domain::CompletionRequest request);

    /**
     * Submit a streaming request with a callback for each chunk.
     */
    void submit_streaming_request(
        domain::CompletionRequest request,
        std::function<void(const domain::StreamingChunkResponse&, bool is_final)> callback
    );

    /**
     * Set the device runner factory.
     */
    void set_runner_factory(std::function<std::unique_ptr<runners::BaseDeviceRunner>(const std::string&)> factory) {
        runner_factory_ = std::move(factory);
    }

    /**
     * Get worker info for monitoring.
     */
    struct WorkerInfo {
        std::string worker_id;
        bool is_ready;
        size_t processed_requests;
    };
    std::vector<WorkerInfo> get_worker_info() const;

private:
    /**
     * Worker thread function.
     */
    void worker_loop(const std::string& worker_id);

    /**
     * Process a single task.
     */
    void process_task(SchedulerTask& task, runners::BaseDeviceRunner& runner);

    ThreadSafeQueue<SchedulerTask> task_queue_;
    std::vector<std::thread> worker_threads_;
    std::unordered_map<std::string, std::unique_ptr<runners::BaseDeviceRunner>> runners_;

    std::function<std::unique_ptr<runners::BaseDeviceRunner>(const std::string&)> runner_factory_;

    std::atomic<bool> is_ready_{false};
    std::atomic<bool> running_{false};

    mutable std::mutex workers_mutex_;
    std::unordered_map<std::string, WorkerInfo> worker_info_;

    int worker_count_ = DEFAULT_WORKER_COUNT;
};

} // namespace tt::scheduler
