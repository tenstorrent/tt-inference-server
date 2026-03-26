// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ipc/boost_ipc_cancel_queue.hpp"
#include "ipc/boost_ipc_task_queue.hpp"
#include "ipc/cancel_queue.hpp"
#include "ipc/shared_memory.hpp"

namespace tt::ipc {

constexpr const char* TASK_QUEUE_NAME = "tt_tasks";
constexpr size_t RING_BUFFER_CAPACITY = 65536;

/**
 * Manages task queue, result queues, and cancel queues for LLM workers.
 * Handles creation, cleanup, and coordination of IPC queues.
 */
class QueueManager {
 public:
  std::shared_ptr<BoostIpcTaskQueue> task_queue;
  std::vector<std::shared_ptr<TokenRingBuffer<RING_BUFFER_CAPACITY>>>
      result_queues;
  std::vector<std::shared_ptr<ICancelQueue>> cancel_queues;

  explicit QueueManager(int numWorkers) {
    BoostIpcTaskQueue::remove(TASK_QUEUE_NAME);
    task_queue = std::make_shared<BoostIpcTaskQueue>(TASK_QUEUE_NAME, 1024);
    result_queues.reserve(numWorkers);
    cancel_queues.reserve(numWorkers);
    for (int i = 0; i < numWorkers; i++) {
      result_queues.emplace_back(
          std::make_shared<TokenRingBuffer<RING_BUFFER_CAPACITY>>(
              "/tt_tokens_" + std::to_string(i), true));

      std::string cancelName =
          std::string(CANCEL_QUEUE_NAME) + "_" + std::to_string(i);
      BoostIpcCancelQueue::removeByName(cancelName);
      cancel_queues.emplace_back(std::make_shared<BoostIpcCancelQueue>(
          cancelName, CANCEL_QUEUE_CAPACITY));
    }
  }

  ~QueueManager() { clear(); }

  void clear() {
    BoostIpcTaskQueue::remove(TASK_QUEUE_NAME);
    for (auto& queue : result_queues) {
      queue->shutdown();
    }
    for (auto& cq : cancel_queues) {
      cq->remove();
    }
    cancel_queues.clear();
  }

  // Delete copy constructor and assignment operator
  QueueManager(const QueueManager&) = delete;
  QueueManager& operator=(const QueueManager&) = delete;

  // Allow move constructor and assignment operator
  QueueManager(QueueManager&&) = default;
  QueueManager& operator=(QueueManager&&) = default;
};

}  // namespace tt::ipc
