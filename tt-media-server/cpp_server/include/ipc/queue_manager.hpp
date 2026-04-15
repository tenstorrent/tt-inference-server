// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ipc/boost_ipc_cancel_queue.hpp"
#include "ipc/boost_ipc_result_queue.hpp"
#include "ipc/boost_ipc_task_queue.hpp"

namespace tt::ipc {

constexpr const char* TASK_QUEUE_NAME = "tt_tasks";
constexpr const char* RESULT_QUEUE_PREFIX = "tt_results_";
constexpr const char* CANCEL_QUEUE_PREFIX = "tt_cancels_";
constexpr size_t CANCEL_QUEUE_CAPACITY = 1024;

/**
 * Manages task queue, result queues, and cancel queues for LLM workers.
 * Handles creation, cleanup, and coordination of IPC queues.
 */
class QueueManager {
 public:
  std::shared_ptr<BoostIpcTaskQueue> task_queue;
  std::vector<std::shared_ptr<BoostIpcResultQueue>> result_queues;
  std::vector<std::shared_ptr<BoostIpcCancelQueue>> cancel_queues;

  explicit QueueManager(int numWorkers) {
    task_queue = std::make_shared<BoostIpcTaskQueue>(TASK_QUEUE_NAME, 1024);
    result_queues.reserve(numWorkers);
    cancel_queues.reserve(numWorkers);
    for (int i = 0; i < numWorkers; i++) {
      std::string resultName =
          std::string(RESULT_QUEUE_PREFIX) + std::to_string(i);
      result_queues.emplace_back(std::make_shared<BoostIpcResultQueue>(
          resultName, RESULT_QUEUE_CAPACITY));

      std::string cancelName =
          std::string(CANCEL_QUEUE_PREFIX) + std::to_string(i);
      cancel_queues.emplace_back(std::make_shared<BoostIpcCancelQueue>(
          cancelName, CANCEL_QUEUE_CAPACITY));
    }
  }

  ~QueueManager() { clear(); }

  void clear() {
    BoostIpcTaskQueue::remove(TASK_QUEUE_NAME);
    for (auto& queue : result_queues) {
      queue->shutdown();
      queue->remove();
    }
    for (size_t i = 0; i < cancel_queues.size(); i++) {
      cancel_queues[i]->remove();
    }
  }

  QueueManager(const QueueManager&) = delete;
  QueueManager& operator=(const QueueManager&) = delete;

  QueueManager(QueueManager&&) = default;
  QueueManager& operator=(QueueManager&&) = default;
};

}  // namespace tt::ipc
