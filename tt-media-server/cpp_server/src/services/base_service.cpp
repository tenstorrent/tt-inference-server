// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "services/base_service.hpp"
#include "runners/runner_factory.hpp"

#include <iostream>

namespace tt::services {

BaseService::BaseService() {
    // Use multiprocess scheduler with 4 workers by default
    // Can be overridden with TT_NUM_WORKERS environment variable
    const char* num_workers_env = std::getenv("TT_NUM_WORKERS");
    size_t num_workers = num_workers_env ? std::stoul(num_workers_env) : 4;

    scheduler_ = std::make_shared<scheduler::MultiprocessScheduler>(num_workers);

    // Set runner factory based on TT_RUNNER_TYPE environment variable
    scheduler_->set_runner_factory(runners::RunnerFactory::get_factory());

    std::cout << "[BaseService] Initialized with MultiprocessScheduler (" << num_workers << " workers)" << std::endl;
}

void BaseService::start() {
    std::cout << "[BaseService] Starting service..." << std::endl;
    scheduler_->start();
}

void BaseService::stop() {
    std::cout << "[BaseService] Stopping service..." << std::endl;
    scheduler_->stop();
}

bool BaseService::is_model_ready() const {
    return scheduler_->is_ready();
}

BaseService::SystemStatus BaseService::get_system_status() const {
    return SystemStatus{
        .model_ready = scheduler_->is_ready(),
        .queue_size = scheduler_->queue_size(),
        .max_queue_size = max_queue_size_,
        .device = device_,
        .worker_info = scheduler_->get_worker_info()
    };
}

std::future<domain::CompletionResponse> BaseService::process_request(domain::CompletionRequest request) {
    // Pre-process
    request = pre_process(std::move(request));

    // Submit to scheduler
    return scheduler_->submit_request(std::move(request));
}

void BaseService::process_streaming_request(
    domain::CompletionRequest request,
    std::function<void(const domain::StreamingChunkResponse&)> chunk_callback,
    std::function<void()> done_callback) {

    // Pre-process
    request = pre_process(std::move(request));

    // Submit streaming request
    scheduler_->submit_streaming_request(
        std::move(request),
        [chunk_callback = std::move(chunk_callback), done_callback = std::move(done_callback)]
        (const domain::StreamingChunkResponse& chunk, bool is_final) {
            chunk_callback(chunk);
            if (is_final) {
                done_callback();
            }
        }
    );
}

} // namespace tt::services
