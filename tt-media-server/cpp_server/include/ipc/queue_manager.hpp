// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include "ipc/shared_memory.hpp"
#include "ipc/boost_ipc_task_queue.hpp"
#include "ipc/cancel_queue.hpp"
#include <memory>
#include <string>
#include <vector>

namespace tt::ipc {

using namespace std;

constexpr const char* TASK_QUEUE_NAME = "tt_tasks";
constexpr size_t RING_BUFFER_CAPACITY = 65536;

/**
 * Manages task queue and result queues for LLM workers.
 * Handles creation, cleanup, and coordination of IPC queues.
 */
class QueueManager {
public:
    shared_ptr<BoostIpcTaskQueue> task_queue;
    vector<shared_ptr<TokenRingBuffer<RING_BUFFER_CAPACITY>>> result_queues;
    shared_ptr<CancelQueue> cancel_queue;

    explicit QueueManager(int num_workers) {
        BoostIpcTaskQueue::remove(TASK_QUEUE_NAME);
        CancelQueue::remove(CANCEL_QUEUE_NAME);
        task_queue = make_shared<BoostIpcTaskQueue>(TASK_QUEUE_NAME, 1024);
        cancel_queue = make_shared<CancelQueue>(CANCEL_QUEUE_NAME, 256);
        result_queues.reserve(num_workers);
        for (int i = 0; i < num_workers; i++) {
            result_queues.emplace_back(make_shared<TokenRingBuffer<RING_BUFFER_CAPACITY>>(
                "/tt_tokens_" + to_string(i), true
            ));
        }
    }
    
    ~QueueManager() {
        clear();
    }
    
    void clear() {
        BoostIpcTaskQueue::remove(TASK_QUEUE_NAME);
        CancelQueue::remove(CANCEL_QUEUE_NAME);
        for (auto& queue : result_queues) {
            queue->shutdown();
        }
    }

    // Delete copy constructor and assignment operator
    QueueManager(const QueueManager&) = delete;
    QueueManager& operator=(const QueueManager&) = delete;

    // Allow move constructor and assignment operator
    QueueManager(QueueManager&&) = default;
    QueueManager& operator=(QueueManager&&) = default;
};

} // namespace tt::ipc
