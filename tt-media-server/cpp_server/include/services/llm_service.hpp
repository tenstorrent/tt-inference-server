// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include "services/base_service.hpp"
#include "ipc/queue_manager.hpp"
#include "worker/single_process_worker.hpp"
#include "config/constants.hpp"
#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"
#include "domain/prefill_request.hpp"
#include "domain/prefill_result.hpp"
#include "services/streamable.hpp"
#include "sockets/inter_server_service.hpp"

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
    : public BaseService<domain::CompletionRequest, domain::CompletionResponse>
    , public Streamable<domain::CompletionRequest, domain::StreamingChunkResponse> {
public:
    // Callback types for transport-agnostic communication
    using PrefillRequestCallback = std::function<bool(const domain::PrefillRequest&)>;
    using PrefillResultCallback = std::function<void(const domain::PrefillResult&)>;

    LLMService();
    ~LLMService() override;

    LLMService(const LLMService&) = delete;
    LLMService& operator=(const LLMService&) = delete;

    void start() override;
    void stop() override;

    bool is_model_ready() const override;
    SystemStatus get_system_status() const override;

    // Handler methods - called by controllers (transport-agnostic)
    void handle_prefill_request(
        const std::string& task_id,
        const std::string& prompt,
        const std::vector<int64_t>& token_ids,
        int max_tokens);
    void handle_prefill_complete(const domain::PrefillResult& result);
    void handle_connection_lost();

    // Set callbacks for transport-agnostic communication (called by controller)
    void set_prefill_request_callback(PrefillRequestCallback callback);
    void set_prefill_result_callback(PrefillResultCallback callback);

    // Get socket service for controller initialization
    std::shared_ptr<tt::sockets::InterServerService> get_socket_service() const;

protected:
    void pre_process(domain::CompletionRequest& request) const override;
    void post_process(domain::CompletionResponse& response) const override;
    domain::CompletionResponse process_request(
        domain::CompletionRequest request) override;

    void streaming_pre_process(domain::CompletionRequest& request) const override { pre_process(request); }
    void streaming_post_process(domain::StreamingChunkResponse&) const override {}
    void process_streaming_request(
        domain::CompletionRequest request,
        std::function<void(domain::StreamingChunkResponse&, bool is_final)> callback
    ) override;

private:
    void start_workers();
    void start_consumers();

    void consumer_loop_for_worker(size_t worker_idx);

    bool check_worker_alive(size_t worker_idx);

    void continue_decode_generation(const domain::PrefillResult& prefill_result);

    tt::config::LLMMode mode_;

    std::vector<std::unique_ptr<worker::SingleProcessWorker>> workers_;
    size_t num_workers_;

    std::vector<std::thread> consumer_threads_;

    ConcurrentMap<std::string, std::function<void(domain::StreamingChunkResponse&, bool)>> stream_callbacks_;

    std::atomic<uint64_t> next_worker_{0};

    std::atomic<size_t> pending_tasks_{0};

    std::atomic<bool> is_ready_{false};
    std::atomic<bool> running_{false};

    size_t max_queue_size_ = 10000;
    std::string device_ = "cpu";

    std::unique_ptr<tt::ipc::QueueManager> queue_manager_;
    tt::utils::Tokenizer tokenizer_;
    std::shared_ptr<tt::sockets::InterServerService> socket_service_;

    // Callbacks for transport-agnostic communication (set by controller)
    PrefillRequestCallback prefill_request_callback_;
    PrefillResultCallback prefill_result_callback_;
};

}
