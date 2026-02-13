// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "services/llm_service.hpp"
#include "config/settings.hpp"
#include "scheduler/multiprocess_scheduler.hpp"
#include "utils/tokenizer.hpp"

#include <chrono>
#include <string>
#include <thread>
#include <utility>
#include <unistd.h>

namespace tt::services {

LLMService::LLMService()
    : num_workers_(tt::config::num_workers()),
      scheduler_(std::make_unique<scheduler::MultiprocessScheduler>(num_workers_)) {}

LLMService::~LLMService() = default;

void LLMService::start() {
    std::string tokenizer_path = tt::config::tokenizer_path();
    if (!tokenizer_path.empty()) {
        scheduler_->set_token_decoder([tokenizer_path](uint64_t token_id) {
            return tt::utils::Tokenizer::instance(tokenizer_path).decode({static_cast<int>(token_id)});
        });
    }
    std::vector<scheduler::MultiprocessScheduler::WorkerEnvConfig> env_configs(num_workers_);
    for (size_t i = 0; i < num_workers_; ++i) {
        env_configs[i].env_vars["TT_VISIBLE_DEVICES"] = tt::config::visible_devices_for_worker(i);
    }
    scheduler_->start(env_configs);
    is_ready_ = true;
    running_ = true;
}

void LLMService::stop() {
    running_ = false;
    is_ready_ = false;
    scheduler_->stop();
}

bool LLMService::is_model_ready() const {
    return scheduler_->is_ready();
}

BaseService::SystemStatus LLMService::get_system_status() const {
    BaseService::SystemStatus status;
    status.model_ready = scheduler_->is_ready();
    status.queue_size = scheduler_->queue_size();
    status.max_queue_size = max_queue_size_;
    status.device = device_;
    auto worker_info = scheduler_->get_worker_info();
    status.worker_info.reserve(worker_info.size());
    for (const auto& w : worker_info) {
        status.worker_info.push_back({
            .worker_id = w.worker_id,
            .is_ready = w.is_ready,
            .processed_requests = w.processed_requests,
        });
    }
    return status;
}

std::future<domain::CompletionResponse> LLMService::process_request(domain::CompletionRequest request) {
    return scheduler_->submit_request(std::move(request));
}

void LLMService::process_streaming_request(
    domain::CompletionRequest request,
    std::function<void(const domain::StreamingChunkResponse&)> chunk_callback,
    std::function<void()> done_callback) {
    scheduler_->submit_streaming_request(
        std::move(request),
        [chunk_callback = std::move(chunk_callback),
         done_callback = std::move(done_callback)](const domain::StreamingChunkResponse& chunk, bool is_final) {
            chunk_callback(chunk);
            if (is_final) {
                done_callback();
            }
        });
}

LLMService::Stats LLMService::get_stats() const {
    auto s = scheduler_->get_stats();
    return {
        .tokens_produced = s.tokens_produced,
        .tokens_consumed = s.tokens_consumed,
        .tasks_submitted = s.tasks_submitted,
        .tasks_completed = s.tasks_completed,
        .avg_token_latency_us = s.avg_token_latency_us,
    };
}

bool LLMService::check_worker_alive(size_t /*worker_idx*/) {
    return running_.load();
}

void LLMService::fail_all_pending(const std::string& /*reason*/) {}

[[noreturn]] void LLMService::worker_process_main(int /*worker_id*/, const WorkerEnvConfig& /*env_config*/) {
    _exit(0);
}

void LLMService::consumer_loop_for_worker(size_t /*worker_idx*/) {}

void LLMService::dispatch_task(ProcessTask /*task*/) {}

void LLMService::submit_streaming_request(
    domain::CompletionRequest /*request*/,
    std::function<void(const domain::StreamingChunkResponse&, bool)> /*callback*/) {}

std::future<domain::CompletionResponse> LLMService::submit_request(domain::CompletionRequest /*request*/) {
    std::promise<domain::CompletionResponse> p;
    p.set_value({});
    return p.get_future();
}

}  // namespace tt::services
