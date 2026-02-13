// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "services/llm_service.hpp"
#include "config/settings.hpp"
#include "ipc/shared_memory.hpp"
#include "runners/llm_engine/config.hpp"
#include "runners/llm_engine/engine/llm_engine.hpp"
#include "runners/llm_engine/engine/scheduler.hpp"
#include "runners/llm_engine/engine/boost_ipc_task_queue.hpp"

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <chrono>
#include <cstring>
#include <csignal>
#include <execinfo.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <sys/wait.h>
#include <unistd.h>

namespace {
void worker_crash_handler(int sig) {
    const char* msg = "\n[Worker CRASH] Signal: ";
    write(STDERR_FILENO, msg, strlen(msg));
    char buf[16];
    int len = snprintf(buf, sizeof(buf), "%d\n", sig);
    write(STDERR_FILENO, buf, len);
    void* frames[64];
    int n = backtrace(frames, 64);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);
    _exit(128 + sig);
}
} // namespace

namespace tt::services {

// ---------------------------------------------------------------------------
// Anonymous helpers
// ---------------------------------------------------------------------------

constexpr const char* TASK_QUEUE_NAME = "tt_tasks";

namespace {
    std::mutex task_id_gen_mutex;
    boost::uuids::random_generator task_id_generator;

    std::string generate_task_id() {
        std::lock_guard<std::mutex> lock(task_id_gen_mutex);
        return boost::uuids::to_string(task_id_generator());
    }
} // anonymous namespace

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

LLMService::LLMService()
    : num_workers_(tt::config::num_workers()) {
    std::cout << "[LLMService] Initialized (" << num_workers_ << " workers)\n" << std::flush;
}

LLMService::~LLMService() {
    stop();
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void LLMService::start() {
    if (running_.exchange(true)) {
        return;  // Already running
    }

    std::cout << "[LLMService] Starting with " << num_workers_ << " worker processes\n" << std::flush;

    workers_.resize(num_workers_);

    llm_engine::BoostIpcTaskQueue::remove(TASK_QUEUE_NAME);
    auto task_buffer = std::make_shared<llm_engine::BoostIpcTaskQueue>(TASK_QUEUE_NAME, 1024);

    // Create shared memory buffers and spawn workers
    for (size_t i = 0; i < num_workers_; i++) {
        auto& worker = workers_[i];
        worker.worker_id = static_cast<int>(i);

        std::string token_shm_name = "/tt_tokens_" + std::to_string(i);

        // Clean up any existing shared memory from previous runs
        shm_unlink(token_shm_name.c_str());

        worker.token_buffer = std::make_unique<ipc::TokenRingBuffer<RING_BUFFER_CAPACITY>>(
            token_shm_name, true  // Create as owner
        );
        worker.task_buffer = task_buffer;

        // Build environment config for this worker
        WorkerEnvConfig env_config;
        env_config.env_vars["TT_VISIBLE_DEVICES"] = tt::config::visible_devices_for_worker(i);

        // Fork worker process
        pid_t pid = fork();

        if (pid < 0) {
            throw std::runtime_error("Failed to fork worker process");
        } else if (pid == 0) {
            // Child process — must never fall through to parent's loop
            try {
                worker_process_main(static_cast<int>(i), env_config);
            } catch (const std::exception& e) {
                std::cerr << "[LLMService] Worker " << i << " failed: " << e.what() << "\n" << std::flush;
            }
            _exit(1);
        } else {
            // Parent process
            worker.pid = pid;
            std::cout << "[LLMService] Spawned worker " << i << " with PID " << pid << "\n" << std::flush;
        }
    }

    // Start one consumer thread per worker for parallel token processing
    consumer_threads_.reserve(num_workers_);
    for (size_t i = 0; i < num_workers_; i++) {
        consumer_threads_.emplace_back(&LLMService::consumer_loop_for_worker, this, i);
    }
    std::cout << "[LLMService] Started " << num_workers_ << " consumer threads\n" << std::flush;

    // Wait for workers to signal ready
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Mark all workers as ready
    for (auto& worker : workers_) {
        worker.is_ready = true;
    }

    is_ready_ = true;
    std::cout << "[LLMService] All workers started\n" << std::flush;
}

void LLMService::stop() {
    if (!running_.exchange(false)) {
        return;
    }

    std::cout << "[LLMService] Stopping...\n" << std::flush;

    // Signal shutdown to all workers
    for (auto& worker : workers_) {
        if (worker.token_buffer) {
            worker.token_buffer->shutdown();
        }
    }

    // Wait for all consumer threads
    for (auto& thread : consumer_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    consumer_threads_.clear();

    // Wait for worker processes
    for (auto& worker : workers_) {
        if (worker.pid > 0) {
            kill(worker.pid, SIGTERM);

            int status;
            int wait_result = waitpid(worker.pid, &status, WNOHANG);
            if (wait_result == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                wait_result = waitpid(worker.pid, &status, WNOHANG);
                if (wait_result == 0) {
                    kill(worker.pid, SIGKILL);
                    waitpid(worker.pid, &status, 0);
                }
            }
            std::cout << "[LLMService] Worker " << worker.worker_id << " exited\n" << std::flush;
        }
    }

    workers_.clear();
    is_ready_ = false;
    std::cout << "[LLMService] Stopped\n" << std::flush;
}

// ---------------------------------------------------------------------------
// Status queries
// ---------------------------------------------------------------------------

bool LLMService::is_model_ready() const {
    return is_ready_.load();
}

BaseService::SystemStatus LLMService::get_system_status() const {
    SystemStatus status;
    status.model_ready = is_ready_.load();
    status.queue_size = pending_tasks_.load();
    status.max_queue_size = max_queue_size_;
    status.device = device_;

    status.worker_info.reserve(workers_.size());
    for (size_t i = 0; i < workers_.size(); i++) {
        status.worker_info.push_back({
            .worker_id = "worker-" + std::to_string(i),
            .is_ready = workers_[i].is_ready && workers_[i].is_alive,
            .processed_requests = 0  // TODO: track per-worker stats
        });
    }

    return status;
}

LLMService::Stats LLMService::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

// ---------------------------------------------------------------------------
// Worker process (child)
// ---------------------------------------------------------------------------

[[noreturn]] void LLMService::worker_process_main(int worker_id, const WorkerEnvConfig& env_config) {
    // === CHILD PROCESS ===

    // Install crash handler so any SIGSEGV prints a backtrace.
    signal(SIGSEGV, worker_crash_handler);
    signal(SIGABRT, worker_crash_handler);
    signal(SIGBUS, worker_crash_handler);

    // 1. Set environment variables BEFORE any device library init
    for (const auto& [key, value] : env_config.env_vars) {
        setenv(key.c_str(), value.c_str(), 1);
    }

    std::cout << "[Worker " << worker_id << "] Started with PID " << getpid() << "\n" << std::flush;
    for (const auto& [key, value] : env_config.env_vars) {
        std::cout << "[Worker " << worker_id << "] ENV: " << key << "=" << value << "\n" << std::flush;
    }

    // 2. Attach to shared memory (don't create, just attach)
    std::string token_shm_name = "/tt_tokens_" + std::to_string(worker_id);

    ipc::TokenRingBuffer<RING_BUFFER_CAPACITY> token_buffer(token_shm_name, false);

    auto task_buffer = std::make_shared<llm_engine::BoostIpcTaskQueue>(TASK_QUEUE_NAME);
    llm_engine::Config config;
    config.num_kvcache_blocks = 128;

    auto scheduler = std::make_unique<llm_engine::Scheduler>(config, task_buffer);

    auto engine = std::make_unique<llm_engine::LLMEngine>(
        config,
        [&token_buffer](llm_engine::SequenceID seq_id, uint64_t token_id, bool finished) {
            auto token = ipc::SharedToken{
                .token_index = 0,
                .flags = static_cast<uint32_t>(finished ? 1 : 0),
                .token_id = token_id,
                .task_id = {},
                .padding = {},
            };
            std::strncpy(token.task_id, seq_id.id.c_str(), sizeof(token.task_id) - 1);
            token.task_id[sizeof(token.task_id) - 1] = '\0';
            token_buffer.push(token);
        },
        std::move(scheduler));
    engine->run();
    _exit(0);
}

// ---------------------------------------------------------------------------
// Consumer loop (parent-side, one thread per worker)
// ---------------------------------------------------------------------------

bool LLMService::check_worker_alive(size_t worker_idx) {
    auto& worker = workers_[worker_idx];
    if (!worker.is_alive || worker.pid <= 0) {
        return false;
    }

    int status = 0;
    pid_t result = waitpid(worker.pid, &status, WNOHANG);
    if (result == 0) {
        return true;  // Still running
    }

    // Worker has exited
    worker.is_alive = false;
    worker.is_ready = false;

    if (result == worker.pid) {
        if (WIFEXITED(status)) {
            std::cerr << "[LLMService] Worker " << worker.worker_id
                      << " (PID " << worker.pid << ") exited with code "
                      << WEXITSTATUS(status) << "\n" << std::flush;
        } else if (WIFSIGNALED(status)) {
            std::cerr << "[LLMService] Worker " << worker.worker_id
                      << " (PID " << worker.pid << ") killed by signal "
                      << WTERMSIG(status) << "\n" << std::flush;
        }
    } else {
        std::cerr << "[LLMService] Worker " << worker.worker_id
                  << " (PID " << worker.pid << ") waitpid error\n" << std::flush;
    }

    fail_all_pending("Worker " + std::to_string(worker.worker_id) + " crashed");
    return false;
}

void LLMService::fail_all_pending(const std::string& reason) {
    // Fail all streaming callbacks
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        for (auto& [task_id, callback] : stream_callbacks_) {
            domain::StreamingChunkResponse error_response;
            error_response.id = "cmpl-" + task_id;

            domain::CompletionChoice choice;
            choice.text = "";
            choice.index = 0;
            choice.finish_reason = "error";
            error_response.choices.push_back(std::move(choice));

            callback(error_response, /*is_final=*/true);
        }
        stream_callbacks_.clear();
    }

    // Fail all non-streaming promises
    {
        std::lock_guard<std::mutex> lock(promises_mutex_);
        for (auto& [task_id, promise] : result_promises_) {
            promise->set_exception(std::make_exception_ptr(
                std::runtime_error(reason)));
        }
        result_promises_.clear();
    }

    pending_tasks_.store(0);
}

void LLMService::consumer_loop_for_worker(size_t worker_idx) {
    std::cout << "[Consumer-" << worker_idx << "] Started\n" << std::flush;

    auto& worker = workers_[worker_idx];
    if (!worker.token_buffer) {
        std::cout << "[Consumer-" << worker_idx << "] No token buffer, exiting\n" << std::flush;
        return;
    }

    while (running_) {
        // Check if the worker process is still alive
        if (!check_worker_alive(worker_idx)) {
            std::cerr << "[Consumer-" << worker_idx << "] Worker process died, exiting consumer\n" << std::flush;
            break;
        }

        bool any_activity = false;

        ipc::SharedToken token;
        while (worker.token_buffer->pop(token)) {
            any_activity = true;

            // Find callback for this task
            std::function<void(const domain::StreamingChunkResponse&, bool)> callback;
            {
                std::lock_guard<std::mutex> lock(callbacks_mutex_);
                auto it = stream_callbacks_.find(token.task_id);
                if (it != stream_callbacks_.end()) {
                    callback = it->second;
                    if (token.is_final()) {
                        stream_callbacks_.erase(it);
                    }
                }
            }

            if (callback) {
                domain::StreamingChunkResponse response;
                response.id = "cmpl-" + std::string(token.task_id);
                response.created = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();

                domain::CompletionChoice choice;
                choice.text = std::to_string(token.token_id);
                choice.index = token.token_index;
                if (token.is_final()) {
                    choice.finish_reason = "stop";
                }
                response.choices.push_back(std::move(choice));

                callback(response, token.is_final());
            }

            // Handle non-streaming completion
            if (token.is_final() && (token.flags & ipc::SharedToken::FLAG_DONE)) {
                std::lock_guard<std::mutex> lock(promises_mutex_);
                auto it = result_promises_.find(std::string(token.task_id));
                if (it != result_promises_.end()) {
                    domain::CompletionResponse response;
                    response.id = "cmpl-" + std::string(token.task_id);
                    it->second->set_value(response);
                    result_promises_.erase(it);
                }
            }
        }

        if (!any_activity) {
            std::this_thread::yield();
        }
    }

    std::cout << "[Consumer-" << worker_idx << "] Stopped\n" << std::flush;
}

// ---------------------------------------------------------------------------
// Task dispatch
// ---------------------------------------------------------------------------

void LLMService::dispatch_task(ProcessTask task) {
    uint64_t worker_idx = next_worker_.fetch_add(1) % workers_.size();
    auto& worker = workers_[worker_idx];

    auto prompt = std::get<std::vector<int>>(task.request.prompt);
    std::vector<int64_t> token_ids(prompt.begin(), prompt.end());
    auto sequence = std::make_unique<llm_engine::Sequence>(token_ids);
    sequence->seq_id.id = task.task_id;
    sequence->num_prompt_tokens_ = prompt.size();
    sequence->temperature = task.request.temperature.value_or(1.0f);
    sequence->max_tokens = task.request.max_tokens;
    sequence->ignore_eos = task.request.ignore_eos;
    worker.task_buffer->push(*sequence);
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.tasks_submitted++;
    }
}

// ---------------------------------------------------------------------------
// Request submission (internal)
// ---------------------------------------------------------------------------

void LLMService::submit_streaming_request(
    domain::CompletionRequest request,
    std::function<void(const domain::StreamingChunkResponse&, bool is_final)> callback) {

    std::string task_id = request.task_id.empty() ? generate_task_id() : request.task_id;
    request.task_id = task_id;

    pending_tasks_.fetch_add(1);

    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        stream_callbacks_[task_id] = [this, cb = std::move(callback)](
            const domain::StreamingChunkResponse& chunk, bool is_final) {
            cb(chunk, is_final);
            if (is_final) {
                pending_tasks_.fetch_sub(1);
            }
        };
    }

    ProcessTask task;
    task.request = std::move(request);
    task.is_streaming = true;
    task.task_id = task_id;

    dispatch_task(std::move(task));
}

std::future<domain::CompletionResponse> LLMService::submit_request(domain::CompletionRequest request) {
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

// ---------------------------------------------------------------------------
// Public request interface (BaseService overrides)
// ---------------------------------------------------------------------------

std::future<domain::CompletionResponse> LLMService::process_request(domain::CompletionRequest request) {
    request = pre_process(std::move(request));
    return submit_request(std::move(request));
}

void LLMService::process_streaming_request(
    domain::CompletionRequest request,
    std::function<void(const domain::StreamingChunkResponse&)> chunk_callback,
    std::function<void()> done_callback) {

    request = pre_process(std::move(request));

    submit_streaming_request(
        std::move(request),
        [chunk_callback = std::move(chunk_callback), done_callback = std::move(done_callback)](
            const domain::StreamingChunkResponse& chunk, bool is_final) {
            chunk_callback(chunk);
            if (is_final) {
                done_callback();
            }
        }
    );
}

} // namespace tt::services
