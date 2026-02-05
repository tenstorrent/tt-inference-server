// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "scheduler/multiprocess_scheduler.hpp"
#include "runners/llm_test_runner.hpp"
#include "runners/runner_factory.hpp"

#include <iostream>
#include <chrono>
#include <cstring>
#include <sys/eventfd.h>
#include <poll.h>

namespace tt::scheduler {

namespace {
    // Helper to generate unique task IDs
    std::atomic<uint64_t> task_id_counter{0};

    std::string generate_task_id() {
        return "mp-task-" + std::to_string(task_id_counter.fetch_add(1));
    }
}

MultiprocessScheduler::MultiprocessScheduler(size_t num_workers)
    : num_workers_(num_workers) {}

MultiprocessScheduler::~MultiprocessScheduler() {
    stop();
}

void MultiprocessScheduler::start(const std::vector<WorkerEnvConfig>& env_configs) {
    if (running_.exchange(true)) {
        return;  // Already running
    }

    std::cout << "[MultiprocessScheduler] Starting with " << num_workers_ << " worker processes\n";

    workers_.resize(num_workers_);

    // Create shared memory buffers and spawn workers
    for (size_t i = 0; i < num_workers_; i++) {
        auto& worker = workers_[i];
        worker.worker_id = static_cast<int>(i);

        // Create shared memory ring buffers
        std::string token_shm_name = "/tt_tokens_" + std::to_string(i);
        std::string task_shm_name = "/tt_tasks_" + std::to_string(i);

        // Clean up any existing shared memory from previous runs
        shm_unlink(token_shm_name.c_str());
        shm_unlink(task_shm_name.c_str());

        worker.token_buffer = std::make_unique<ipc::TokenRingBuffer<RING_BUFFER_CAPACITY>>(
            token_shm_name, true  // Create as owner
        );
        worker.task_buffer = std::make_unique<ipc::TokenRingBuffer<1024>>(
            task_shm_name, true
        );

        // Get environment config for this worker
        WorkerEnvConfig env_config;
        if (i < env_configs.size()) {
            env_config = env_configs[i];
        }
        // Always set device ID
        env_config.env_vars["TT_DEVICE_ID"] = std::to_string(i);
        env_config.env_vars["TT_WORKER_ID"] = std::to_string(i);

        // Fork worker process
        pid_t pid = fork();

        if (pid < 0) {
            throw std::runtime_error("Failed to fork worker process");
        } else if (pid == 0) {
            // Child process
            worker_process_main(static_cast<int>(i), env_config);
            // Never returns
        } else {
            // Parent process
            worker.pid = pid;
            std::cout << "[MultiprocessScheduler] Spawned worker " << i << " with PID " << pid << "\n";
        }
    }

    // Start consumer thread
    consumer_thread_ = std::thread(&MultiprocessScheduler::consumer_loop, this);

    // Wait for workers to signal ready
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Mark all workers as ready
    for (auto& worker : workers_) {
        worker.is_ready = true;
    }

    is_ready_ = true;
    std::cout << "[MultiprocessScheduler] All workers started\n";
}

void MultiprocessScheduler::stop() {
    if (!running_.exchange(false)) {
        return;
    }

    std::cout << "[MultiprocessScheduler] Stopping...\n";

    // Signal shutdown to all workers
    for (auto& worker : workers_) {
        if (worker.token_buffer) {
            worker.token_buffer->shutdown();
        }
        if (worker.task_buffer) {
            worker.task_buffer->shutdown();
        }
    }

    // Wait for consumer thread
    if (consumer_thread_.joinable()) {
        consumer_thread_.join();
    }

    // Wait for worker processes
    for (auto& worker : workers_) {
        if (worker.pid > 0) {
            // Send SIGTERM first
            kill(worker.pid, SIGTERM);

            // Wait with timeout
            int status;
            int wait_result = waitpid(worker.pid, &status, WNOHANG);
            if (wait_result == 0) {
                // Process still running, wait a bit
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                wait_result = waitpid(worker.pid, &status, WNOHANG);
                if (wait_result == 0) {
                    // Still running, force kill
                    kill(worker.pid, SIGKILL);
                    waitpid(worker.pid, &status, 0);
                }
            }
            std::cout << "[MultiprocessScheduler] Worker " << worker.worker_id << " exited\n";
        }
    }

    workers_.clear();
    is_ready_ = false;
    std::cout << "[MultiprocessScheduler] Stopped\n";
}

[[noreturn]] void MultiprocessScheduler::worker_process_main(int worker_id, const WorkerEnvConfig& env_config) {
    // === CHILD PROCESS ===

    // 1. Set environment variables BEFORE any device library init
    for (const auto& [key, value] : env_config.env_vars) {
        setenv(key.c_str(), value.c_str(), 1);
    }

    std::cout << "[Worker " << worker_id << "] Started with PID " << getpid() << "\n";
    for (const auto& [key, value] : env_config.env_vars) {
        std::cout << "[Worker " << worker_id << "] ENV: " << key << "=" << value << "\n";
    }

    // 2. Attach to shared memory (don't create, just attach)
    std::string token_shm_name = "/tt_tokens_" + std::to_string(worker_id);
    std::string task_shm_name = "/tt_tasks_" + std::to_string(worker_id);

    ipc::TokenRingBuffer<RING_BUFFER_CAPACITY> token_buffer(token_shm_name, false);
    ipc::TokenRingBuffer<1024> task_buffer(task_shm_name, false);

    // 3. Create the device runner using factory (reads TT_RUNNER_TYPE from environment)
    auto runner = runners::RunnerFactory::create("device_" + std::to_string(worker_id));
    runner->warmup();

    std::cout << "[Worker " << worker_id << "] Ready\n";

    // 4. Main work loop
    while (!task_buffer.is_shutdown()) {
        // Check for new task
        ipc::SharedToken task_token;
        if (!task_buffer.pop(task_token)) {
            // No task available, brief sleep
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }

        // Parse task from token
        // For simplicity, we encode: task_id in task_id field, max_tokens in token_index
        std::string task_id(task_token.task_id);
        int max_tokens = static_cast<int>(task_token.token_index);
        bool is_streaming = !(task_token.flags & ipc::SharedToken::FLAG_DONE);

        std::cout << "[Worker " << worker_id << "] Processing task " << task_id
                  << " with " << max_tokens << " tokens\n";

        // Build request
        domain::CompletionRequest request;
        request.task_id = task_id;
        request.max_tokens = max_tokens;
        request.stream = is_streaming;

        if (is_streaming) {
            // Run streaming inference, push tokens to shared memory
            auto chunk_callback = [&](const domain::StreamingChunkOutput& chunk) {
                ipc::SharedToken out_token;
                out_token.worker_id = worker_id;
                out_token.token_index = chunk.chunk.index.value_or(0);
                out_token.flags = 0;

                // Copy task_id
                std::strncpy(out_token.task_id, chunk.task_id.c_str(), sizeof(out_token.task_id) - 1);
                out_token.task_id[sizeof(out_token.task_id) - 1] = '\0';

                // Copy text
                std::strncpy(out_token.text, chunk.chunk.text.c_str(), sizeof(out_token.text) - 1);
                out_token.text[sizeof(out_token.text) - 1] = '\0';

                // Push to ring buffer (spin if full)
                while (!token_buffer.push(out_token) && !token_buffer.is_shutdown()) {
                    std::this_thread::yield();
                }
            };

            auto final_callback = [&](const domain::FinalResultOutput& result) {
                ipc::SharedToken out_token;
                out_token.worker_id = worker_id;
                out_token.token_index = result.result.index.value_or(0);
                out_token.flags = ipc::SharedToken::FLAG_FINAL;
                if (result.result.finish_reason == "error") {
                    out_token.flags |= ipc::SharedToken::FLAG_ERROR;
                }

                std::strncpy(out_token.task_id, result.task_id.c_str(), sizeof(out_token.task_id) - 1);
                out_token.task_id[sizeof(out_token.task_id) - 1] = '\0';
                std::strncpy(out_token.text, result.result.text.c_str(), sizeof(out_token.text) - 1);
                out_token.text[sizeof(out_token.text) - 1] = '\0';

                while (!token_buffer.push(out_token) && !token_buffer.is_shutdown()) {
                    std::this_thread::yield();
                }
            };

            runner->run_streaming(request, chunk_callback, final_callback);
        } else {
            // Non-streaming: run and send single result
            std::vector<domain::CompletionRequest> requests = {request};
            auto responses = runner->run(requests);

            // Send response as final token
            ipc::SharedToken out_token;
            out_token.worker_id = worker_id;
            out_token.flags = ipc::SharedToken::FLAG_FINAL | ipc::SharedToken::FLAG_DONE;
            std::strncpy(out_token.task_id, task_id.c_str(), sizeof(out_token.task_id) - 1);

            while (!token_buffer.push(out_token) && !token_buffer.is_shutdown()) {
                std::this_thread::yield();
            }
        }
    }

    std::cout << "[Worker " << worker_id << "] Shutting down\n";
    _exit(0);
}

void MultiprocessScheduler::consumer_loop() {
    std::cout << "[Consumer] Started\n";

    // Poll all worker token buffers
    while (running_) {
        bool any_activity = false;

        for (auto& worker : workers_) {
            if (!worker.token_buffer) continue;

            ipc::SharedToken token;
            while (worker.token_buffer->pop(token)) {
                any_activity = true;

                std::string task_id(token.task_id);

                // Update stats
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.tokens_consumed++;
                }

                // Find callback for this task
                std::function<void(const domain::StreamingChunkResponse&, bool)> callback;
                {
                    std::lock_guard<std::mutex> lock(callbacks_mutex_);
                    auto it = stream_callbacks_.find(task_id);
                    if (it != stream_callbacks_.end()) {
                        callback = it->second;
                        if (token.is_final()) {
                            stream_callbacks_.erase(it);
                        }
                    }
                }

                if (callback) {
                    // Build response
                    domain::StreamingChunkResponse response;
                    response.id = "cmpl-" + task_id;
                    response.created = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count();

                    domain::CompletionChoice choice;
                    choice.text = token.text;
                    choice.index = token.token_index;
                    if (token.is_final()) {
                        choice.finish_reason = "stop";
                    }
                    response.choices.push_back(choice);

                    // Call the callback
                    callback(response, token.is_final());
                }

                // Handle non-streaming completion
                if (token.is_final() && (token.flags & ipc::SharedToken::FLAG_DONE)) {
                    std::lock_guard<std::mutex> lock(promises_mutex_);
                    auto it = result_promises_.find(task_id);
                    if (it != result_promises_.end()) {
                        domain::CompletionResponse response;
                        response.id = "cmpl-" + task_id;
                        it->second->set_value(response);
                        result_promises_.erase(it);
                    }

                    std::lock_guard<std::mutex> slock(stats_mutex_);
                    stats_.tasks_completed++;
                }
            }
        }

        if (!any_activity) {
            // No tokens available, brief sleep
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    std::cout << "[Consumer] Stopped\n";
}

void MultiprocessScheduler::dispatch_task(ProcessTask task) {
    // Round-robin dispatch
    uint64_t worker_idx = next_worker_.fetch_add(1) % workers_.size();
    auto& worker = workers_[worker_idx];

    // Encode task as SharedToken
    ipc::SharedToken task_token;
    task_token.worker_id = worker_idx;
    task_token.token_index = task.request.max_tokens;  // Encode max_tokens
    task_token.flags = task.is_streaming ? 0 : ipc::SharedToken::FLAG_DONE;

    std::strncpy(task_token.task_id, task.task_id.c_str(), sizeof(task_token.task_id) - 1);
    task_token.task_id[sizeof(task_token.task_id) - 1] = '\0';

    // Push to worker's task buffer
    while (!worker.task_buffer->push(task_token) && running_) {
        std::this_thread::yield();
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.tasks_submitted++;
    }
}

void MultiprocessScheduler::submit_streaming_request(
    domain::CompletionRequest request,
    std::function<void(const domain::StreamingChunkResponse&, bool is_final)> callback) {

    std::string task_id = request.task_id.empty() ? generate_task_id() : request.task_id;
    request.task_id = task_id;

    // Track pending tasks
    pending_tasks_.fetch_add(1);

    // Register callback (wraps original to decrement pending count)
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        stream_callbacks_[task_id] = [this, cb = std::move(callback)](const domain::StreamingChunkResponse& chunk, bool is_final) {
            cb(chunk, is_final);
            if (is_final) {
                pending_tasks_.fetch_sub(1);
            }
        };
    }

    // Dispatch task
    ProcessTask task;
    task.request = std::move(request);
    task.is_streaming = true;
    task.task_id = task_id;

    dispatch_task(std::move(task));
}

std::future<domain::CompletionResponse> MultiprocessScheduler::submit_request(domain::CompletionRequest request) {
    std::string task_id = request.task_id.empty() ? generate_task_id() : request.task_id;
    request.task_id = task_id;

    auto promise = std::make_shared<std::promise<domain::CompletionResponse>>();
    auto future = promise->get_future();

    {
        std::lock_guard<std::mutex> lock(promises_mutex_);
        result_promises_[task_id] = promise;
    }

    ProcessTask task;
    task.request = std::move(request);
    task.is_streaming = false;
    task.task_id = task_id;

    dispatch_task(std::move(task));

    return future;
}

MultiprocessScheduler::Stats MultiprocessScheduler::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

std::vector<MultiprocessScheduler::WorkerInfo> MultiprocessScheduler::get_worker_info() const {
    std::vector<WorkerInfo> result;
    result.reserve(workers_.size());

    for (size_t i = 0; i < workers_.size(); i++) {
        result.push_back({
            .worker_id = "worker-" + std::to_string(i),
            .is_ready = workers_[i].is_ready,
            .processed_requests = 0  // TODO: track per-worker stats
        });
    }

    return result;
}

} // namespace tt::scheduler
