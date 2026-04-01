// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ipc/boost_ipc_cancel_queue.hpp"
#include "ipc/boost_ipc_task_queue.hpp"
#include "ipc/token_ring_buffer.hpp"

namespace tt::ipc {

using namespace std;

constexpr const char* TASK_QUEUE_NAME = "tt_tasks";
constexpr const char* CANCEL_QUEUE_PREFIX = "tt_cancels_";
constexpr size_t RING_BUFFER_CAPACITY = 65536;
constexpr size_t CANCEL_QUEUE_CAPACITY = 1024;

/**
 * Manages task queue, result queues, and cancel queues for LLM workers.
 * Handles creation, cleanup, and coordination of IPC queues.
 */
class QueueManager {
 public:
  shared_ptr<BoostIpcTaskQueue> task_queue;
  vector<shared_ptr<TokenRingBuffer<RING_BUFFER_CAPACITY>>> result_queues;
  vector<shared_ptr<BoostIpcCancelQueue>> cancel_queues;

  explicit QueueManager(int numWorkers) {
    BoostIpcTaskQueue::remove(TASK_QUEUE_NAME);
    task_queue = make_shared<BoostIpcTaskQueue>(TASK_QUEUE_NAME, 1024);
    result_queues.reserve(numWorkers);
    cancel_queues.reserve(numWorkers);
    for (int i = 0; i < numWorkers; i++) {
      result_queues.emplace_back(
          make_shared<TokenRingBuffer<RING_BUFFER_CAPACITY>>(
              "/tt_tokens_" + to_string(i), true));

      string cancelName = CANCEL_QUEUE_PREFIX + to_string(i);
      BoostIpcCancelQueue::removeByName(cancelName);
      cancel_queues.emplace_back(
          make_shared<BoostIpcCancelQueue>(cancelName, CANCEL_QUEUE_CAPACITY));
    }
  }

  ~QueueManager() { clear(); }

  void clear() {
    BoostIpcTaskQueue::remove(TASK_QUEUE_NAME);
    for (auto& queue : result_queues) {
      queue->shutdown();
    }
    for (size_t i = 0; i < cancel_queues.size(); i++) {
      cancel_queues[i]->remove();
    }
  }

  // Delete copy constructor and assignment operator
  QueueManager(const QueueManager&) = delete;
  QueueManager& operator=(const QueueManager&) = delete;

  // Allow move constructor and assignment operator
  QueueManager(QueueManager&&) = default;
  QueueManager& operator=(QueueManager&&) = default;
};

}  // namespace tt::ipc
