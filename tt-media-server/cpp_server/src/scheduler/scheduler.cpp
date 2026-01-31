// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "scheduler/scheduler.hpp"
#include "runners/llm_test_runner.hpp"

#include <iostream>
#include <sstream>

namespace tt::scheduler {

Scheduler::Scheduler()
    : task_queue_(DEFAULT_QUEUE_SIZE)
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

    // Wait for all worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();

    // Clean up runners
    runners_.clear();

    {
        std::lock_guard<std::mutex> lock(workers_mutex_);
        worker_info_.clear();
    }

    std::cout << "[Scheduler] All workers stopped" << std::endl;
}

std::future<domain::CompletionResponse> Scheduler::submit_request(domain::CompletionRequest request) {
    auto promise = std::make_shared<std::promise<domain::CompletionResponse>>();
    auto future = promise->get_future();

    SchedulerTask task;
    task.request = std::move(request);
    task.is_streaming = false;
    task.result_promise = promise;

    if (!task_queue_.push(std::move(task))) {
        promise->set_exception(std::make_exception_ptr(
            std::runtime_error("Failed to submit request: queue shutdown")
        ));
    }

    return future;
}

void Scheduler::submit_streaming_request(
    domain::CompletionRequest request,
    std::function<void(const domain::StreamingChunkResponse&, bool is_final)> callback) {

    SchedulerTask task;
    task.request = std::move(request);
    task.is_streaming = true;
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
        process_task(task, *runner);

        // Update stats
        {
            std::lock_guard<std::mutex> lock(workers_mutex_);
            worker_info_[worker_id].processed_requests++;
        }
    }

    runner->close();
    std::cout << "[Worker " << worker_id << "] Stopped" << std::endl;
}

void Scheduler::process_task(SchedulerTask& task, runners::BaseDeviceRunner& runner) {
    try {
        if (task.is_streaming) {
            // Streaming request
            auto created = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();

            std::string completion_id = "cmpl-" + task.request.task_id;
            std::string model = task.request.model.value_or("test-model");

            runner.run_streaming(
                task.request,
                // Chunk callback
                [&task, &completion_id, &model, created](const domain::StreamingChunkOutput& chunk) {
                    domain::StreamingChunkResponse response;
                    response.id = completion_id;
                    response.created = created;
                    response.model = model;

                    domain::CompletionChoice choice;
                    choice.text = chunk.chunk.text;
                    choice.index = chunk.chunk.index.value_or(0);
                    choice.finish_reason = chunk.chunk.finish_reason;
                    response.choices.push_back(choice);

                    task.stream_callback(response, false);
                },
                // Final callback
                [&task, &completion_id, &model, created](const domain::FinalResultOutput& final_result) {
                    domain::StreamingChunkResponse response;
                    response.id = completion_id;
                    response.created = created;
                    response.model = model;

                    domain::CompletionChoice choice;
                    choice.text = "";
                    choice.index = 0;
                    choice.finish_reason = "stop";
                    response.choices.push_back(choice);

                    task.stream_callback(response, true);
                }
            );
        } else {
            // Non-streaming request
            auto responses = runner.run({task.request});
            if (!responses.empty()) {
                task.result_promise->set_value(responses[0]);
            } else {
                task.result_promise->set_exception(std::make_exception_ptr(
                    std::runtime_error("No response generated")
                ));
            }
        }
    } catch (const std::exception& e) {
        if (task.result_promise) {
            task.result_promise->set_exception(std::current_exception());
        }
        std::cerr << "[Scheduler] Error processing task: " << e.what() << std::endl;
    }
}

} // namespace tt::scheduler
