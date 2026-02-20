// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/llm_service.hpp"
#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "worker/single_process_worker.hpp"

#include <cassert>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sys/wait.h>

namespace tt::services {

namespace {

[[noreturn]] void exec_worker_process(
    size_t worker_id,
    const std::unordered_map<std::string, std::string>& env_vars) {
    for (const auto& [key, value] : env_vars) {
        setenv(key.c_str(), value.c_str(), 1);
    }
    char exe_path[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (n <= 0) {
        perror("readlink /proc/self/exe");
        _exit(1);
    }
    exe_path[n] = '\0';
    char id_buf[16];
    std::snprintf(id_buf, sizeof(id_buf), "%zu", worker_id);
    char* exec_argv[] = {exe_path, const_cast<char*>("--worker"), id_buf, nullptr};
    execv(exe_path, exec_argv);
    perror("execv");
    _exit(1);
}

}  // namespace

worker::WorkerConfig make_worker_config_for_process(int worker_id) {
    worker::WorkerConfig cfg;
    cfg.env_vars["TT_VISIBLE_DEVICES"] = tt::config::visible_devices_for_worker(worker_id);
    cfg.task_queue = std::make_shared<tt::ipc::BoostIpcTaskQueue>(tt::ipc::TASK_QUEUE_NAME);
    cfg.result_queue = std::make_shared<tt::ipc::TokenRingBuffer<tt::ipc::RING_BUFFER_CAPACITY>>(
        "/tt_tokens_" + std::to_string(worker_id), false);
    cfg.worker_id = worker_id;
    return cfg;
}

LLMService::LLMService()
    : num_workers_(tt::config::num_workers()),
      tokenizer_(tt::config::tokenizer_path()) {
    std::cout << "[LLMService] Initialized (" << num_workers_ << " workers)\n" << std::flush;
    queue_manager_ = std::make_unique<tt::ipc::QueueManager>(num_workers_);
}

LLMService::~LLMService() {
    stop();
}

void LLMService::start() {
    ZoneScopedN("LLMService::start");
    if (running_.exchange(true)) {
        return;  // Already running
    }

    std::cout << "[LLMService] Starting with " << num_workers_ << " worker processes\n" << std::flush;
    start_workers();
    tracy_config::TracyStartupSchedulerParent();
    start_consumers();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    is_ready_ = true;
    TracyPlot("pending_tasks", static_cast<double>(pending_tasks_.load()));
    std::cout << "[LLMService] Service started\n" << std::flush;
}

bool LLMService::is_model_ready() const {
    return is_ready_.load();
}

SystemStatus LLMService::get_system_status() const {
    SystemStatus status;
    status.model_ready = is_ready_.load();
    status.queue_size = pending_tasks_.load();
    status.max_queue_size = max_queue_size_;
    status.device = device_;

    for (const auto& w : workers_) {
        WorkerInfo info;
        info.worker_id = std::to_string(w->worker_id);
        info.is_ready = w->is_ready;
        info.processed_requests = 0;  // TODO: track per-worker stats
        status.worker_info.push_back(info);
    }
    return status;
}

void LLMService::pre_process(domain::CompletionRequest& request) const {
    if (std::holds_alternative<std::string>(request.prompt)) {
        auto text = std::get<std::string>(request.prompt);
        request.prompt = tokenizer_.encode(text);
    }
}


void LLMService::start_workers() {
    auto create_worker_config = [this](int worker_id) -> tt::worker::WorkerConfig {
        return {
            .env_vars = {
                {"TT_VISIBLE_DEVICES", tt::config::visible_devices_for_worker(worker_id)}
            },
            .task_queue = queue_manager_->task_queue,
            .result_queue = queue_manager_->result_queues[worker_id],
            .worker_id = worker_id
        };
    };

    for (size_t i = 0; i < num_workers_; i++) {
        auto cfg = create_worker_config(static_cast<int>(i));
        workers_.push_back(std::make_unique<tt::worker::SingleProcessWorker>(cfg));
        auto& worker = workers_[i];

        pid_t pid = fork();

        if (pid < 0) {
            throw std::runtime_error("Failed to fork worker process");
        }
        if (pid == 0) {
            try {
                exec_worker_process(i, cfg.env_vars);
            } catch (const std::exception& e) {
                std::cerr << "[LLMService] Worker " << i << " failed: " << e.what() << "\n" << std::flush;
                _exit(1);
            }
        }
        worker->pid = pid;
        std::cout << "[LLMService] Spawned worker " << i << " with PID " << pid << "\n" << std::flush;
    }
}

void LLMService::start_consumers() {
    consumer_threads_.reserve(num_workers_);
    for (size_t i = 0; i < num_workers_; i++) {
        consumer_threads_.emplace_back(&LLMService::consumer_loop_for_worker, this, i);
    }
    std::cout << "[LLMService] Started " << num_workers_ << " consumer threads\n" << std::flush;
}

void LLMService::stop() {
    ZoneScopedN("LLMService::stop");
    if (!running_.exchange(false)) {
        return;
    }

    std::cout << "[LLMService] Stopping...\n" << std::flush;


    // Wait for all consumer threads
    for (auto& thread : consumer_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    consumer_threads_.clear();

    // Signal shutdown to all workers
    for (auto& w : workers_) {
        w->stop();
    }

    workers_.clear();
    is_ready_ = false;
    std::cout << "[LLMService] Stopped\n" << std::flush;
    queue_manager_->clear();
}

bool LLMService::check_worker_alive(size_t worker_idx) {
    auto* worker = workers_[worker_idx].get();
    if (worker->pid <= 0) {
        return false;
    }

    int status;
    pid_t result = waitpid(worker->pid, &status, WNOHANG);
    if (result == 0) {
        return true;  // Still running
    }
    if (result == worker->pid) {
        worker->is_alive = false;
        return false;
    }
    return true;  // Error in waitpid, assume alive
}

void LLMService::consumer_loop_for_worker(size_t worker_idx) {
    ZoneScopedN("LLMService::consumer_loop");
    tracy_config::TracySetThreadName(
        ("Consumer-" + std::to_string(worker_idx)).c_str());

    std::cout << "[Consumer-" << worker_idx << "] Started\n" << std::flush;

    auto* worker = workers_[worker_idx].get();
    if (!worker->cfg.result_queue) {
        std::cout << "[Consumer-" << worker_idx << "] No token buffer, exiting\n" << std::flush;
        return;
    }

    tt::utils::Tokenizer tokenizer(tt::config::tokenizer_path());

    while (running_) {
        if (!check_worker_alive(worker_idx)) {
            std::cerr << "[Consumer-" << worker_idx << "] Worker process died, exiting consumer\n" << std::flush;
            break;
        }

        bool any_activity = false;

        ipc::SharedToken token;
        while (worker->cfg.result_queue->pop(token)) {
            any_activity = true;

            auto val = stream_callbacks_.get(token.task_id);
            if (!val.has_value()) {
                throw std::runtime_error("callback not found for task_id: " + std::string(token.task_id));
            }
            auto callback = val.value();
            if (token.is_final()) {
                stream_callbacks_.erase(token.task_id);
            }

            domain::StreamingChunkResponse response;
            response.id = std::string(token.task_id);
            response.created = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();

            domain::CompletionChoice choice;
            choice.index = token.token_index;
            if (token.is_error()) {
                choice.finish_reason = "error";
            } else {
                choice.text = token.is_stop_token() ? "" : tokenizer.decode({static_cast<int>(token.token_id)});
                if (token.is_final()) {
                    choice.finish_reason = token.is_stop_token() ? "stop" : "length";
                }
            }
            response.choices.push_back(std::move(choice));

            callback(response, token.is_final());
            if (token.is_final()) {
                TracyPlot("pending_tasks", static_cast<double>(pending_tasks_.load()));
            }
        }

        if (!any_activity) {
            std::this_thread::yield();
        }
    }

    std::cout << "[Consumer-" << worker_idx << "] Stopped\n" << std::flush;
}

domain::CompletionResponse LLMService::process_request(domain::CompletionRequest request) {
    ZoneScopedN("LLMService::process_request");

    std::mutex mtx;
    std::condition_variable cv;
    bool done = false;

    std::string accumulated_text;
    int completion_tokens = 0;
    std::string finish_reason = "stop";

    const int prompt_tokens = std::holds_alternative<std::vector<int>>(request.prompt)
        ? static_cast<int>(std::get<std::vector<int>>(request.prompt).size())
        : 0;
    const std::string task_id = request.task_id;
    const std::string model = request.model.value_or("default");

    process_streaming_request(std::move(request),
        [&](domain::StreamingChunkResponse& chunk, bool is_final) {
            if (!chunk.choices.empty()) {
                accumulated_text.append(chunk.choices[0].text);
                completion_tokens++;
                if (chunk.choices[0].finish_reason.has_value()) {
                    finish_reason = chunk.choices[0].finish_reason.value();
                }
            }
            if (is_final) {
                std::lock_guard<std::mutex> lock(mtx);
                done = true;
                cv.notify_one();
            }
        });

    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&] { return done; });

    domain::CompletionResponse response;
    response.id = task_id;
    response.model = model;
    response.created = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    domain::CompletionChoice choice;
    choice.text = std::move(accumulated_text);
    choice.index = 0;
    choice.finish_reason = finish_reason;
    response.choices.push_back(std::move(choice));

    response.usage = {prompt_tokens, completion_tokens, prompt_tokens + completion_tokens};

    return response;
}

void LLMService::process_streaming_request(
    domain::CompletionRequest request,
    std::function<void(domain::StreamingChunkResponse&, bool is_final)> callback) {
    assert(callback != nullptr);

    ZoneScopedN("LLMService::process_streaming_request");
    if (request.task_id.empty()) {
        throw std::runtime_error("task_id must be set before submitting request");
    }
    std::string task_id = request.task_id;

    pending_tasks_.fetch_add(1);
    TracyPlot("pending_tasks", static_cast<double>(pending_tasks_.load()));

    stream_callbacks_.insert(task_id, [this, cb = std::move(callback)](
        domain::StreamingChunkResponse& chunk, bool is_final) {
        cb(chunk, is_final);
        if (is_final) {
            pending_tasks_.fetch_sub(1);
        }
    });

    auto prompt = std::get<std::vector<int>>(request.prompt);
    std::vector<int64_t> token_ids(prompt.begin(), prompt.end());
    auto sequence = std::make_unique<llm_engine::Sequence>(token_ids);
    sequence->task_id.id = task_id;
    sequence->num_prompt_tokens_ = prompt.size();
    sequence->temperature = request.temperature.value_or(1.0f);
    sequence->max_tokens = request.max_tokens;
    sequence->ignore_eos = request.ignore_eos;
    if (request.seed.has_value()) {
      sequence->seed = request.seed;
    }
    queue_manager_->task_queue->push(*std::move(sequence));
}

void LLMService::post_process(domain::CompletionResponse&) const {
    // no-op
}

}
