// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/llm_service.hpp"
#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "utils/tokenizer.hpp"
#include "utils/logger.hpp"
#include "worker/single_process_worker.hpp"
#include "utils/mapper.hpp"
#include <cassert>
#include <unordered_set>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstring>
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
    cfg.runner_config = tt::config::llm_engine_config();
    return cfg;
}

LLMService::LLMService()
    : mode_(tt::config::llm_mode()),
      num_workers_(tt::config::num_workers()),
      tokenizer_(&tt::utils::active_tokenizer()) {
    max_queue_size_ = tt::config::max_queue_size();
    TT_LOG_INFO("[LLMService] Initialized (mode={}, workers={})",
                tt::config::to_string(mode_), num_workers_);
    queue_manager_ = std::make_unique<tt::ipc::QueueManager>(num_workers_);

    socket_service_ = std::make_shared<tt::sockets::InterServerService>();
    socket_service_->initializeFromConfig();
}

LLMService::~LLMService() {
    stop();
}

void LLMService::start() {
    ZoneScopedN("LLMService::start");
    if (running_.exchange(true)) {
        return;  // Already running
    }

    TT_LOG_INFO("[LLMService] Starting (mode={}, workers={})",
                tt::config::to_string(mode_), num_workers_);

    start_workers();
    tracy_config::TracyStartupSchedulerParent();
    start_consumers();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (socket_service_ && socket_service_->isEnabled()) {
        socket_service_->start();
    }

    is_ready_ = true;
    TracyPlot("pending_tasks", static_cast<double>(pending_tasks_.load()));
    TT_LOG_INFO("[LLMService] Service started");
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

size_t LLMService::current_queue_size() const {
    return pending_tasks_.load();
}

void LLMService::pre_process(domain::CompletionRequest& request) const {
    BaseService::pre_process(request);
    if (std::holds_alternative<std::string>(request.prompt)) {
        auto text = std::get<std::string>(request.prompt);
        static auto cfg = tt::utils::get_tokenizer_config();
        bool has_bos = text.size() >= cfg.bos_token.size() &&
                       text.compare(0, cfg.bos_token.size(), cfg.bos_token) == 0;
        if (cfg.add_bos_token && !cfg.bos_token.empty() && !has_bos) {
            text = cfg.bos_token + text;
        }
        request.prompt = tokenizer_->encode(text);
    }
    const auto& tokens = std::get<std::vector<int>>(request.prompt);
    if (tokens.size() > tt::config::LLMConfig::MAX_INPUT_TOKENS) {
        throw std::invalid_argument(
            "Input too long: " + std::to_string(tokens.size()) +
            " tokens exceeds maximum of " + std::to_string(tt::config::LLMConfig::MAX_INPUT_TOKENS));
    }
    // Set prompt token count after tokenization
    request.prompt_tokens_count = static_cast<int>(tokens.size());
}

void LLMService::start_workers() {
    for (size_t i = 0; i < num_workers_; i++) {
        tt::worker::WorkerConfig cfg = make_worker_config_for_process(static_cast<int>(i));
        workers_.push_back(std::make_unique<tt::worker::SingleProcessWorker>(cfg));
        auto& worker = workers_[i];

        pid_t pid = fork();

        if (pid < 0) {
            throw std::runtime_error("Failed to fork worker process");
        }
        if (pid == 0) {
            setpgid(0, 0);
            try {
                exec_worker_process(i, cfg.env_vars);
            } catch (const std::exception& e) {
                TT_LOG_ERROR("[LLMService] Worker {} failed: {}", i, e.what());
                _exit(1);
            }
        }
        setpgid(pid, pid);
        worker->pid = pid;
        TT_LOG_INFO("[LLMService] Spawned worker {} with PID {}", i, pid);
    }
}

void LLMService::start_consumers() {
    consumer_threads_.reserve(num_workers_);
    for (size_t i = 0; i < num_workers_; i++) {
        consumer_threads_.emplace_back(&LLMService::consumer_loop_for_worker, this, i);
    }
    TT_LOG_INFO("[LLMService] Started {} consumer threads", num_workers_);
}

void LLMService::stop() {
    ZoneScopedN("LLMService::stop");
    if (!running_.exchange(false)) {
        return;
    }

    TT_LOG_INFO("[LLMService] Stopping...");

    // Signal shutdown on all ring buffers so blocking_pop wakes up
    for (auto& q : queue_manager_->result_queues) {
        q->shutdown();
    }

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

    // Stop socket service
    if (socket_service_) {
        socket_service_->stop();
    }

    is_ready_ = false;
    TT_LOG_INFO("[LLMService] Stopped");
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

    TT_LOG_INFO("[Consumer-{}] Started", worker_idx);

    auto* worker = workers_[worker_idx].get();
    if (!worker->cfg.result_queue) {
        TT_LOG_WARN("[Consumer-{}] No token buffer, exiting", worker_idx);
        return;
    }

    const auto stop_ids = tokenizer_->stop_token_ids();
    const std::unordered_set<int64_t> stop_token_set(stop_ids.begin(), stop_ids.end());

    while (running_) {
        if (!check_worker_alive(worker_idx)) {
            TT_LOG_ERROR("[Consumer-{}] Worker process died, exiting consumer", worker_idx);
            break;
        }

        bool any_activity = false;

        ipc::SharedToken token;
        while (worker->cfg.result_queue->blocking_pop(token)) {
            any_activity = true;

            auto val = stream_callbacks_.get(token.task_id);
            if (!val.has_value()) {
                throw std::runtime_error("callback not found for task_id: " + std::string(token.task_id));
            }
            auto callback = val.value();
            if (token.is_final()) {
                stream_callbacks_.erase(token.task_id);
                pending_tasks_.fetch_sub(1);
            }

            domain::StreamingChunkResponse response(
                domain::TaskID(std::string(token.task_id)));
            response.id = std::string(token.task_id);
            response.created = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();

            domain::CompletionChoice choice;
            choice.text = tokenizer_->decode({static_cast<int>(token.token_id)});
            choice.index = token.token_index;
            if (token.is_error()) {
                choice.finish_reason = "error";
            } else {
                choice.token_id = static_cast<int64_t>(token.token_id);
                if (token.is_final()) {
                    bool is_stop = stop_token_set.count(static_cast<int64_t>(token.token_id)) > 0;
                    choice.finish_reason = is_stop ? "stop" : "length";
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

    TT_LOG_INFO("[Consumer-{}] Stopped", worker_idx);
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
    const std::string task_id = request.task_id.id;
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

    domain::CompletionResponse response{domain::TaskID(task_id)};
    response.id = task_id;
    response.model = model;
    response.created = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    domain::CompletionChoice choice;
    choice.text = std::move(accumulated_text);
    choice.index = 0;
    choice.finish_reason = finish_reason;
    response.choices.push_back(std::move(choice));

    response.usage = {prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, std::nullopt, std::nullopt};

    return response;
}

void LLMService::process_streaming_request(
    domain::CompletionRequest request,
    std::function<void(domain::StreamingChunkResponse&, bool is_final)> callback) {
    assert(callback != nullptr);

    ZoneScopedN("LLMService::process_streaming_request");
    if (request.task_id.id.empty()) {
        throw std::runtime_error("task_id must be set before submitting request");
    }
    std::string task_id = request.task_id.id;

    pending_tasks_.fetch_add(1);
    TracyPlot("pending_tasks", static_cast<double>(pending_tasks_.load()));

    stream_callbacks_.insert(task_id, std::move(callback));

    auto prompt = std::get<std::vector<int>>(request.prompt);
    std::vector<int64_t> token_ids(prompt.begin(), prompt.end());

    if (mode_ == tt::config::LLMMode::DECODE_ONLY) {
        if (!prefill_request_callback_) {
            stream_callbacks_.erase(task_id);
            pending_tasks_.fetch_sub(1);
            throw std::runtime_error("No prefill request callback configured");
        }

        domain::PrefillRequest prefill_req{domain::TaskID(task_id)};
        prefill_req.token_ids = token_ids;
        prefill_req.max_tokens = request.max_tokens;

        bool sent = prefill_request_callback_(prefill_req);

        if (!sent) {
            stream_callbacks_.erase(task_id);
            pending_tasks_.fetch_sub(1);
            throw std::runtime_error("Failed to send prefill request (not connected)");
        }
        TT_LOG_DEBUG("[LLMService:DECODE] Forwarded prefill request {} ({} tokens)",
                     task_id, token_ids.size());
        return;
    }

    auto sequence = std::make_unique<llm_engine::Sequence>(
        llm_engine::TaskID(task_id),
        tt::config::llm_engine_config().kvcache_block_size, std::move(token_ids));
    sequence->num_prompt_tokens_ = prompt.size();
    sequence->sampling_params = std::make_unique<llm_engine::SamplingParams>(tt::utils::mapper::map_sampling_params(request));
    queue_manager_->task_queue->push(*std::move(sequence));
}

void LLMService::post_process(domain::CompletionResponse&) const {
    // no-op
}

std::shared_ptr<tt::sockets::InterServerService> LLMService::get_socket_service() const {
    return socket_service_;
}

void LLMService::set_prefill_request_callback(PrefillRequestCallback callback) {
    prefill_request_callback_ = std::move(callback);
}

std::optional<LLMService::StreamCallback> LLMService::detach_stream_callback(const std::string& task_id) {
    auto val = stream_callbacks_.take(task_id);
    if (val.has_value()) {
        pending_tasks_.fetch_sub(1);
    }
    return val;
}

void LLMService::submit_decode_continuation(
    domain::CompletionRequest request, StreamCallback callback) {
    std::string task_id = request.task_id.id;

    pending_tasks_.fetch_add(1);
    TracyPlot("pending_tasks", static_cast<double>(pending_tasks_.load()));
    stream_callbacks_.insert(task_id, std::move(callback));

    auto prompt = std::get<std::vector<int>>(request.prompt);
    std::vector<int64_t> token_ids(prompt.begin(), prompt.end());

    auto sequence = std::make_unique<llm_engine::Sequence>(
        llm_engine::TaskID(task_id),
        tt::config::llm_engine_config().kvcache_block_size, std::move(token_ids));
    sequence->num_prompt_tokens_ = prompt.size();
    sequence->sampling_params = std::make_unique<llm_engine::SamplingParams>(
        tt::utils::mapper::map_sampling_params(request));
    queue_manager_->task_queue->push(*std::move(sequence));

    TT_LOG_DEBUG("[LLMService:DECODE] Queued decode continuation for task {} "
                 "(prompt_tokens={}, max_tokens={})",
                 task_id, prompt.size(),
                 request.max_tokens.has_value()
                     ? std::to_string(request.max_tokens.value())
                     : "none");
}

void LLMService::handle_connection_lost() {
    TT_LOG_ERROR("[LLMService] Failing pending tasks due to connection loss");

    stream_callbacks_.for_each([](const std::string& task_id,
                                   std::function<void(domain::StreamingChunkResponse&, bool)>& callback) {
        domain::StreamingChunkResponse error_response{domain::TaskID(task_id)};
        error_response.id = task_id;
        error_response.created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

        domain::CompletionChoice choice;
        choice.text = "";
        choice.index = 0;
        choice.finish_reason = "error";
        error_response.choices.push_back(std::move(choice));
        error_response.error = "Connection to remote server lost";

        callback(error_response, true);
    });

    stream_callbacks_.clear();
    pending_tasks_.store(0);
}

}
