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

#include "scheduler/thread_safe_queue.hpp"
#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"
#include "runners/base_device_runner.hpp"

namespace tt::scheduler {

/**
 * Task wrapper containing request and metadata.
 * Results are pushed to result_queue_ instead of using callbacks/promises directly.
 */
struct SchedulerTask {
    domain::CompletionRequest request;
    bool is_streaming;

    // Task ID for routing results back to the requester
    std::string task_id;
};

/**
 * Result item that goes into the result queue.
 * Similar to Python's (worker_id, result_key, input_data) tuple.
 */
struct ResultQueueItem {
    std::string worker_id;
    std::string task_id;
    
    // The actual result - either a streaming chunk or final result
    std::variant<
        domain::StreamingChunkResponse,      // Streaming chunk
        domain::CompletionResponse,          // Non-streaming response
        std::string                          // Error message
    > data;
    
    bool is_final;  // True if this is the final result for this task
    bool is_error;  // True if this is an error
};

/**
 * Scheduler manages device workers and request queuing.
 * Similar to the Python Scheduler class.
 * 
 * Architecture:
 * - task_queue_: incoming requests go here
 * - result_queue_: workers put results here (worker_id, task_id, data)
 * - result_listener thread: reads from result_queue_ and dispatches to registered callbacks
 * - result_callbacks_: map of task_id -> callback for streaming results
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
     * The callback receives (response, is_final).
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
     * Worker thread function - processes tasks and puts results in result_queue_.
     */
    void worker_loop(const std::string& worker_id);

    /**
     * Result listener thread function - reads from result_queue_ and dispatches to callbacks.
     * Similar to Python's result_listener().
     */
    void result_listener_loop();

    /**
     * Process a single task and put results in result_queue_.
     */
    void process_task(SchedulerTask& task, runners::BaseDeviceRunner& runner, const std::string& worker_id);

    // Task queue - incoming requests
    ThreadSafeQueue<SchedulerTask> task_queue_;
    
    // Result queue - workers put (worker_id, task_id, data) here
    // Similar to Python's result_queues_by_worker
    ThreadSafeQueue<ResultQueueItem> result_queue_;

    // Worker threads
    std::vector<std::thread> worker_threads_;
    
    // Result listener thread
    std::thread result_listener_thread_;
    
    // Runners (one per worker)
    std::unordered_map<std::string, std::unique_ptr<runners::BaseDeviceRunner>> runners_;

    // Runner factory
    std::function<std::unique_ptr<runners::BaseDeviceRunner>(const std::string&)> runner_factory_;

    // Callbacks for streaming results - map of task_id -> callback
    // Similar to Python's result_queues dict
    mutable std::mutex callbacks_mutex_;
    std::unordered_map<std::string, std::function<void(const domain::StreamingChunkResponse&, bool)>> streaming_callbacks_;
    
    // Promises for non-streaming results
    std::unordered_map<std::string, std::shared_ptr<std::promise<domain::CompletionResponse>>> result_promises_;

    std::atomic<bool> is_ready_{false};
    std::atomic<bool> running_{false};

    mutable std::mutex workers_mutex_;
    std::unordered_map<std::string, WorkerInfo> worker_info_;

    int worker_count_ = DEFAULT_WORKER_COUNT;
};

} // namespace tt::scheduler
