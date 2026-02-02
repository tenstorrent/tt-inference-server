// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "scheduler/scheduler.hpp"
#include "runners/llm_test_runner.hpp"

#include <iostream>
#include <sstream>

namespace tt::scheduler {

Scheduler::Scheduler()
    : task_queue_(DEFAULT_QUEUE_SIZE)
    , result_queue_(DEFAULT_QUEUE_SIZE)
    , is_ready_(false)
    , running_(false) {
    std::cout << "[Scheduler] Initialized with queue size " << DEFAULT_QUEUE_SIZE << std::endl;
}

Scheduler::~Scheduler() {
    stop();
}

void Scheduler::start() {
    if (running_.exchange(true)) {
        return; // Already running
    }

    std::cout << "[Scheduler] Starting " << worker_count_ << " worker(s)..." << std::endl;

    // Start result listener thread first
    result_listener_thread_ = std::thread(&Scheduler::result_listener_loop, this);

    // Create and start worker threads
    for (int i = 0; i < worker_count_; ++i) {
        std::string worker_id = "worker_" + std::to_string(i);

        {
            std::lock_guard<std::mutex> lock(workers_mutex_);
            worker_info_[worker_id] = WorkerInfo{worker_id, false, 0};
        }

        worker_threads_.emplace_back(&Scheduler::worker_loop, this, worker_id);
    }

    // Wait for all workers to be ready
    bool all_ready = false;
    while (!all_ready && running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        std::lock_guard<std::mutex> lock(workers_mutex_);
        all_ready = true;
        for (const auto& [id, info] : worker_info_) {
            if (!info.is_ready) {
                all_ready = false;
                break;
            }
        }
    }

    is_ready_ = true;
    std::cout << "[Scheduler] All workers ready" << std::endl;
}

void Scheduler::stop() {
    if (!running_.exchange(false)) {
        return; // Already stopped
    }

    std::cout << "[Scheduler] Stopping workers..." << std::endl;

    is_ready_ = false;
    task_queue_.shutdown();
    result_queue_.shutdown();

    // Wait for all worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();

    // Wait for result listener thread
    if (result_listener_thread_.joinable()) {
        result_listener_thread_.join();
    }

    // Clean up runners
    runners_.clear();

    // Clean up promises
    {
        std::lock_guard<std::mutex> lock(promises_mutex_);
        result_promises_.clear();
    }

    {
        std::lock_guard<std::mutex> lock(workers_mutex_);
        worker_info_.clear();
    }

    std::cout << "[Scheduler] All workers stopped" << std::endl;
}

std::future<domain::CompletionResponse> Scheduler::submit_request(domain::CompletionRequest request) {
    auto promise = std::make_shared<std::promise<domain::CompletionResponse>>();
    auto future = promise->get_future();

    std::string task_id = request.task_id;

    // Register promise for this task
    {
        std::lock_guard<std::mutex> lock(promises_mutex_);
        result_promises_[task_id] = promise;
    }

    SchedulerTask task;
    task.request = std::move(request);
    task.is_streaming = false;
    task.task_id = task_id;

    if (!task_queue_.push(std::move(task))) {
        // Remove promise on failure
        {
            std::lock_guard<std::mutex> lock(promises_mutex_);
            result_promises_.erase(task_id);
        }
        promise->set_exception(std::make_exception_ptr(
            std::runtime_error("Failed to submit request: queue shutdown")
        ));
    }

    return future;
}

void Scheduler::submit_streaming_request(
    domain::CompletionRequest request,
    std::function<void(const domain::StreamingChunkResponse&, bool is_final)> callback) {

    std::string task_id = request.task_id;

    // Callback is stored directly in the task - worker calls it directly
    // No result queue for streaming = maximum throughput
    SchedulerTask task;
    task.request = std::move(request);
    task.is_streaming = true;
    task.task_id = task_id;
    task.stream_callback = std::move(callback);

    if (!task_queue_.push(std::move(task))) {
        throw std::runtime_error("Failed to submit streaming request: queue shutdown");
    }
}

std::vector<Scheduler::WorkerInfo> Scheduler::get_worker_info() const {
    std::lock_guard<std::mutex> lock(workers_mutex_);
    std::vector<WorkerInfo> result;
    result.reserve(worker_info_.size());
    for (const auto& [id, info] : worker_info_) {
        result.push_back(info);
    }
    return result;
}

void Scheduler::result_listener_loop() {
    std::cout << "[ResultListener] Starting (non-streaming only)..." << std::endl;

    while (running_) {
        auto item_opt = result_queue_.pop();
        if (!item_opt.has_value()) {
            break; // Queue shutdown
        }

        const auto& item = item_opt.value();

        // Only handles non-streaming results (streaming bypasses result queue)
        if (auto* response_ptr = std::get_if<domain::CompletionResponse>(&item.data)) {
            // Non-streaming response - set the promise value
            std::shared_ptr<std::promise<domain::CompletionResponse>> promise;
            {
                std::lock_guard<std::mutex> lock(promises_mutex_);
                auto it = result_promises_.find(item.task_id);
                if (it != result_promises_.end()) {
                    promise = it->second;
                    result_promises_.erase(it);
                }
            }
            if (promise) {
                promise->set_value(*response_ptr);
            } else {
                std::cerr << "[ResultListener] No promise found for task " << item.task_id << std::endl;
            }
        } else if (auto* error_ptr = std::get_if<std::string>(&item.data)) {
            // Error for non-streaming
            std::shared_ptr<std::promise<domain::CompletionResponse>> promise;
            {
                std::lock_guard<std::mutex> lock(promises_mutex_);
                auto it = result_promises_.find(item.task_id);
                if (it != result_promises_.end()) {
                    promise = it->second;
                    result_promises_.erase(it);
                }
            }
            if (promise) {
                promise->set_exception(std::make_exception_ptr(
                    std::runtime_error(*error_ptr)
                ));
            } else {
                std::cerr << "[ResultListener] No handler found for error task " << item.task_id << std::endl;
            }
        }
    }

    std::cout << "[ResultListener] Stopped" << std::endl;
}

void Scheduler::worker_loop(const std::string& worker_id) {
    std::cout << "[Worker " << worker_id << "] Starting..." << std::endl;

    // Create runner for this worker
    std::unique_ptr<runners::BaseDeviceRunner> runner;
    if (runner_factory_) {
        runner = runner_factory_(worker_id);
    } else {
        // Default to test runner
        runner = std::make_unique<runners::LLMTestRunner>(worker_id);
    }

    // Warmup
    if (!runner->warmup()) {
        std::cerr << "[Worker " << worker_id << "] Warmup failed!" << std::endl;
        return;
    }

    // Mark as ready
    {
        std::lock_guard<std::mutex> lock(workers_mutex_);
        worker_info_[worker_id].is_ready = true;
    }
    std::cout << "[Worker " << worker_id << "] Ready" << std::endl;

    // Process tasks
    while (running_) {
        auto task_opt = task_queue_.pop();
        if (!task_opt.has_value()) {
            break; // Queue shutdown
        }

        auto& task = task_opt.value();
        process_task(task, *runner, worker_id);

        // Update stats
        {
            std::lock_guard<std::mutex> lock(workers_mutex_);
            worker_info_[worker_id].processed_requests++;
        }
    }

    runner->close();
    std::cout << "[Worker " << worker_id << "] Stopped" << std::endl;
}

void Scheduler::process_task(SchedulerTask& task, runners::BaseDeviceRunner& runner, const std::string& worker_id) {
    try {
        if (task.is_streaming) {
            // Streaming request - call callback DIRECTLY (no result queue)
            // This is critical for high throughput
            auto created = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();

            std::string completion_id = "cmpl-" + task.request.task_id;
            std::string model = task.request.model.value_or("test-model");

            // Get callback from task (set during submit_streaming_request)
            auto& callback = task.stream_callback;

            runner.run_streaming(
                task.request,
                // Chunk callback - call directly, no queue
                [&callback, &completion_id, &model, created](const domain::StreamingChunkOutput& chunk) {
                    domain::StreamingChunkResponse response;
                    response.id = completion_id;
                    response.created = created;
                    response.model = model;

                    domain::CompletionChoice choice;
                    choice.text = chunk.chunk.text;
                    choice.index = chunk.chunk.index.value_or(0);
                    choice.finish_reason = chunk.chunk.finish_reason;
                    response.choices.push_back(choice);

                    // Call callback directly - no queue overhead!
                    callback(response, false);
                },
                // Final callback - call directly
                [&callback, &completion_id, &model, created](const domain::FinalResultOutput& final_result) {
                    domain::StreamingChunkResponse response;
                    response.id = completion_id;
                    response.created = created;
                    response.model = model;

                    domain::CompletionChoice choice;
                    choice.text = "";
                    choice.index = 0;
                    choice.finish_reason = "stop";
                    response.choices.push_back(choice);

                    // Call callback directly
                    callback(response, true);
                }
            );
        } else {
            // Non-streaming request - push result to result_queue_
            auto responses = runner.run({task.request});
            if (!responses.empty()) {
                ResultQueueItem item;
                item.worker_id = worker_id;
                item.task_id = task.task_id;
                item.data = responses[0];
                item.is_error = false;
                result_queue_.push(std::move(item));
            } else {
                ResultQueueItem item;
                item.worker_id = worker_id;
                item.task_id = task.task_id;
                item.data = std::string("No response generated");
                item.is_error = true;
                result_queue_.push(std::move(item));
            }
        }
    } catch (const std::exception& e) {
        if (task.is_streaming && task.stream_callback) {
            // Send error via callback
            domain::StreamingChunkResponse error_response;
            error_response.id = "error-" + task.task_id;
            error_response.error = e.what();
            task.stream_callback(error_response, true);
        } else {
            // Push error to result queue
            ResultQueueItem item;
            item.worker_id = worker_id;
            item.task_id = task.task_id;
            item.data = std::string(e.what());
            item.is_error = true;
            result_queue_.push(std::move(item));
        }
        std::cerr << "[Scheduler] Error processing task: " << e.what() << std::endl;
    }
}

} // namespace tt::scheduler
