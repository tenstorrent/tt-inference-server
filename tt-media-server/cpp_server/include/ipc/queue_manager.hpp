// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "config/settings.hpp"
#include "ipc/boost_ipc_cancel_queue.hpp"
#include "ipc/boost_ipc_result_queue.hpp"
#include "ipc/boost_ipc_task_queue.hpp"
namespace tt::ipc {

constexpr size_t RING_BUFFER_CAPACITY = 65536;
constexpr size_t CANCEL_QUEUE_CAPACITY = 1024;

/**
 * Manages task queue, result queues, and cancel queues for LLM workers.
 * Handles creation, cleanup, and coordination of IPC queues.
 */
class QueueManager {
 public:
  std::shared_ptr<BoostIpcTaskQueue> taskQueue;
  std::vector<std::shared_ptr<BoostIpcResultQueue>> resultQueues;
  std::vector<std::shared_ptr<BoostIpcCancelQueue>> cancelQueues;

  explicit QueueManager(int numWorkers) {
    taskQueue = std::make_shared<BoostIpcTaskQueue>(
        tt::config::ttTaskQueueName(), 1024);
    resultQueues.reserve(numWorkers);
    cancelQueues.reserve(numWorkers);
    for (int i = 0; i < numWorkers; i++) {
      std::string resultName =
          std::string(tt::config::ttResultQueueName()) + std::to_string(i);
      resultQueues.emplace_back(std::make_shared<BoostIpcResultQueue>(
          resultName, RESULT_QUEUE_CAPACITY));

      std::string cancelName =
          tt::config::ttCancelQueueName() + std::to_string(i);
      cancelQueues.emplace_back(std::make_shared<BoostIpcCancelQueue>(
          cancelName, CANCEL_QUEUE_CAPACITY));
    }
  }

  ~QueueManager() { clear(); }

  void clear() {
    BoostIpcTaskQueue::remove(tt::config::ttTaskQueueName());
    for (auto& queue : resultQueues) {
      queue->shutdown();
      queue->remove();
    }
    for (size_t i = 0; i < cancelQueues.size(); i++) {
      cancelQueues[i]->remove();
    }
  }

  QueueManager(const QueueManager&) = delete;
  QueueManager& operator=(const QueueManager&) = delete;

  QueueManager(QueueManager&&) = default;
  QueueManager& operator=(QueueManager&&) = default;
};

}  // namespace tt::ipc
