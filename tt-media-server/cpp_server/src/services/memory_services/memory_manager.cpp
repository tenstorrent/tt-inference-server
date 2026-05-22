// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/memory_services/memory_manager.hpp"

#include <string>
#include <utility>

#include "config/settings.hpp"
#include "domain/slot_types.hpp"
#include "ipc/boost/boost_memory_queue.hpp"
#include "utils/logger.hpp"

namespace tt::services {

namespace {

class SharedMemoryRequestQueue : public ipc::IMemoryRequestQueue {
 public:
  explicit SharedMemoryRequestQueue(const std::string& name)
      : queue(ipc::boost::MemoryRequestQueue::openExisting(name)) {}

  void push(const domain::ManageMemoryTask& task) override { queue->push(task); }

  bool tryPop(domain::ManageMemoryTask& out) override { return queue->tryPop(out); }

 private:
  std::unique_ptr<ipc::boost::MemoryRequestQueue> queue;
};

class SharedMemoryResultQueue : public ipc::IMemoryResultQueue {
 public:
  explicit SharedMemoryResultQueue(const std::string& name)
      : queue(ipc::boost::MemoryResultQueue::openExisting(name)) {}

  void push(const domain::ManageMemoryResult& result) override {
    queue->push(result);
  }

  bool waitPop(domain::ManageMemoryResult& out) override {
    queue->receive(out);
    return true;
  }

 private:
  std::unique_ptr<ipc::boost::MemoryResultQueue> queue;
};

}  // namespace

MemoryManager::MemoryManager()
    : MemoryManager(
          std::make_shared<SharedMemoryRequestQueue>(
              tt::config::ttMemoryRequestQueueName()),
          std::make_shared<SharedMemoryResultQueue>(
              tt::config::ttMemoryResultQueueName())) {}

MemoryManager::MemoryManager(
    std::shared_ptr<ipc::IMemoryRequestQueue> requestQueue,
    std::shared_ptr<ipc::IMemoryResultQueue> resultQueue)
    : requestQueue(std::move(requestQueue)), resultQueue(std::move(resultQueue)) {
  if (!this->requestQueue || !this->resultQueue) {
    TT_LOG_ERROR(
        "[MemoryManager] Failed to open memory queues. SessionManager should "
        "have created them.");
    throw std::runtime_error("Memory queues not available");
  }

  TT_LOG_INFO("[MemoryManager] Opened memory management queues");
}

MemoryManager::~MemoryManager() {
  TT_LOG_INFO("[MemoryManager] Shutting down");
}

std::optional<domain::ManageMemoryTask> MemoryManager::getRequest() {
  domain::ManageMemoryTask task{};
  if (requestQueue->tryPop(task)) {
    return task;
  }
  return std::nullopt;
}

void MemoryManager::replyAllocateSuccess(uint32_t taskId, uint32_t slotId) {
  domain::ManageMemoryResult result{};
  result.taskId = taskId;
  result.status = domain::ManageMemoryStatus::SUCCESS;
  result.slotId = slotId;
  resultQueue->push(result);
}

void MemoryManager::replyAllocateFailure(uint32_t taskId) {
  domain::ManageMemoryResult result{};
  result.taskId = taskId;
  result.status = domain::ManageMemoryStatus::FAILURE;
  result.slotId = tt::domain::INVALID_SLOT_ID;
  resultQueue->push(result);
}

}  // namespace tt::services
