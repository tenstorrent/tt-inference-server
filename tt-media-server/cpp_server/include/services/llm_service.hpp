// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include "services/base_service.hpp"
#include "ipc/shared_memory.hpp"
#include "worker/base_worker.hpp"
#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"
#include "runners/llm_engine/engine/boost_ipc_task_queue.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include "utils/concurrent_map.hpp"
#include "utils/tokenizer.hpp"
#include <vector>

namespace tt::services {

using namespace std;
    
constexpr const char* TASK_QUEUE_NAME = "tt_tasks";
    
constexpr size_t RING_BUFFER_CAPACITY = 65536;
struct QueueManager {
    shared_ptr<llm_engine::BoostIpcTaskQueue> task_queue;
    vector<shared_ptr<ipc::TokenRingBuffer<RING_BUFFER_CAPACITY>>> result_queues;

    QueueManager(int num_workers) {
        llm_engine::BoostIpcTaskQueue::remove(TASK_QUEUE_NAME);
        task_queue = make_shared<llm_engine::BoostIpcTaskQueue>(TASK_QUEUE_NAME, 1024);
        result_queues.reserve(num_workers);
        for (int i = 0; i < num_workers; i++) {
            result_queues.emplace_back(make_shared<ipc::TokenRingBuffer<RING_BUFFER_CAPACITY>>(
                "/tt_tokens_" + to_string(i), true
            ));
        }
    }
    
    void clear() {
        llm_engine::BoostIpcTaskQueue::remove(TASK_QUEUE_NAME);
        for (auto& queue : result_queues) {
            queue->shutdown();
        }
    }
};
class LLMService : public BaseService {
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
    void post_process(domain::CompletionRequest& request) const override;

    void process_request(
        domain::CompletionRequest request,
        function<void(const domain::StreamingChunkResponse&, bool is_final)> callback
    ) override;
private:
    void start_workers();
    void start_consumers();

    void consumer_loop_for_worker(size_t worker_idx);

    bool check_worker_alive(size_t worker_idx);

    vector<unique_ptr<worker::BaseWorker>> workers_;
    size_t num_workers_;

    vector<thread> consumer_threads_;

    ConcurrentMap<string, function<void(const domain::StreamingChunkResponse&, bool)>> stream_callbacks_;

    atomic<uint64_t> next_worker_{0};

    atomic<size_t> pending_tasks_{0};

    atomic<bool> is_ready_{false};
    atomic<bool> running_{false};

    size_t max_queue_size_ = 10000;
    string device_ = "cpu";

    unique_ptr<QueueManager> queue_manager_;
    tt::utils::Tokenizer tokenizer_;
};

}
