// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include "services/base_service.hpp"
#include "ipc/queue_manager.hpp"
#include "worker/base_worker.hpp"
#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include "utils/concurrent_map.hpp"
#include "utils/tokenizer.hpp"
#include <vector>

namespace tt::services {

worker::WorkerConfig make_worker_config_for_process(int worker_id);

class LLMService
    : public BaseService<domain::CompletionRequest, domain::StreamingChunkResponse>
    , public Streamable<domain::CompletionRequest, domain::StreamingChunkResponse> {
public:

    LLMService();
    ~LLMService() override;

    LLMService(const LLMService&) = delete;
    LLMService& operator=(const LLMService&) = delete;

    void start() override;
    void stop() override;

    bool is_model_ready() const override;
    SystemStatus get_system_status() const override;

protected:
    void pre_process(domain::CompletionRequest& request) const override;
    void post_process(domain::StreamingChunkResponse& response) const override;
    domain::StreamingChunkResponse process_request(
        domain::CompletionRequest request) override;

    void streaming_pre_process(domain::CompletionRequest& request) const override { pre_process(request); }
    void streaming_post_process(domain::StreamingChunkResponse& response) const override { post_process(response); }
    void process_streaming_request(
        domain::CompletionRequest request,
        std::function<void(const domain::StreamingChunkResponse&, bool is_final)> callback
    ) override;

private:
    void start_workers();
    void start_consumers();

    void consumer_loop_for_worker(size_t worker_idx);

    bool check_worker_alive(size_t worker_idx);

    std::vector<std::unique_ptr<worker::BaseWorker>> workers_;
    size_t num_workers_;

    std::vector<std::thread> consumer_threads_;

    ConcurrentMap<std::string, std::function<void(const domain::StreamingChunkResponse&, bool)>> stream_callbacks_;

    std::atomic<uint64_t> next_worker_{0};

    std::atomic<size_t> pending_tasks_{0};

    std::atomic<bool> is_ready_{false};
    std::atomic<bool> running_{false};

    size_t max_queue_size_ = 10000;
    std::string device_ = "cpu";

    std::unique_ptr<tt::ipc::QueueManager> queue_manager_;
    tt::utils::Tokenizer tokenizer_;
};

}
