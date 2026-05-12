#include "services/memory_services/blaze_memory_manager.hpp"

#include "runners/blaze_runner/blaze_utils.hpp"
#include "utils/logger.hpp"

namespace tt::services {
namespace utils = tt::runners::blaze_utils;

BlazeMemoryManager::BlazeMemoryManager(
    tt_blaze::pipeline_manager::PipelineManager& pipelineManager,
    onEvictCb onEvict)
    : pipelineManager(pipelineManager), onEvict(onEvict) {}

std::optional<domain::ManageMemoryTask> BlazeMemoryManager::getRequest() {
  if (!pendingRetries.empty()) {
    auto task = std::move(pendingRetries.front());
    pendingRetries.pop_front();
    return task;
  }
  return MemoryManager::getRequest();
}

void BlazeMemoryManager::handleRequest(
    const domain::ManageMemoryTask& request) {
  switch (request.action) {
    case domain::MemoryManagementAction::ALLOCATE: {
      if (!pipelineManager.push_request(
              utils::makeAllocateRequest(request.taskId))) {
        TT_LOG_DEBUG(
            "[BlazeMemoryManager] ALLOCATE push_request failed; deferring "
            "retry for taskId={}",
            request.taskId);
        pendingRetries.push_back(request);
        return;
      }
      allocating.insert(request.taskId);
      TT_LOG_DEBUG(
          "[BlazeMemoryManager] ALLOCATE: taskId={},"
          "pending allocations={}",
          request.taskId, allocating.size());
      break;
    }
    case domain::MemoryManagementAction::EVICT: {
      auto slotId = request.slotId;
      if (!pipelineManager.push_request(
              utils::makeEvictRequest(request.taskId, slotId))) {
        TT_LOG_DEBUG(
            "[BlazeMemoryManager] EVICT push_request failed for "
            "slotId={}; deferring retry for taskId={}",
            slotId, request.taskId);
        pendingRetries.push_back(request);
        return;
      }
      evicting.insert({request.taskId, slotId});
      TT_LOG_DEBUG(
          "[BlazeMemoryManager] EVICT: taskId={}, slotId={}, "
          "pending evictions={}",
          request.taskId, slotId, evicting.size());
      break;
    }
    case domain::MemoryManagementAction::MOVE: {
      throw std::runtime_error("MOVE action not supported");
    }
    default: {
      throw std::runtime_error("Invalid action");
    }
  }
}

void BlazeMemoryManager::handleResponse(uint32_t taskId, uint32_t slotId) {
  if (allocating.erase(taskId)) {
    TT_LOG_DEBUG(
        "[BlazeMemoryManager] handleResponse[ALLOCATE]: taskId={}, "
        "slotId={}, remaining pending={}",
        taskId, slotId, allocating.size());
    domain::ManageMemoryResult result;
    result.taskId = taskId;
    if (slotId == tt_blaze::pipeline_manager::INVALID_SLOT) {
      TT_LOG_DEBUG(
          "[BlazeMemoryManager] handleResponse[ALLOCATE]: FAILURE "
          "No slot available for taskId={}",
          taskId);
      result.status = domain::ManageMemoryStatus::FAILURE;
      result.slotId = tt_blaze::pipeline_manager::INVALID_SLOT;
      resultQueue->push(result);
      return;
    }
    TT_LOG_DEBUG(
        "[BlazeMemoryManager] handleResponse[ALLOCATE]: SUCCESS taskId={}, "
        "slotId={}",
        taskId, slotId);
    result.status = domain::ManageMemoryStatus::SUCCESS;
    result.slotId = slotId;
    resultQueue->push(result);
    return;
  }
  if (auto it = evicting.find(taskId); it != evicting.end()) {
    auto recordedSlotId = it->second;
    evicting.erase(it);
    if (slotId != recordedSlotId) {
      TT_LOG_ERROR(
          "[BlazeMemoryManager] handleResponse[EVICT]: taskId={} "
          "ack slotId={} does not match recorded slotId={}; evicting "
          "the recorded slot",
          taskId, slotId, recordedSlotId);
    }
    TT_LOG_DEBUG(
        "[BlazeMemoryManager] handleResponse[EVICT]: taskId={}, "
        "slotId={}, remaining pending evictions={}",
        taskId, recordedSlotId, evicting.size());
    onEvict(recordedSlotId);
    return;
  }
  TT_LOG_WARN(
      "[BlazeMemoryManager] handleResponse: unknown taskId={}, slotId={}",
      taskId, slotId);
}

void BlazeMemoryManager::notifyAllocateCancelled(uint32_t taskId) {
  allocating.erase(taskId);
  domain::ManageMemoryResult result;
  result.taskId = taskId;
  result.status = domain::ManageMemoryStatus::FAILURE;
  result.slotId = tt_blaze::pipeline_manager::INVALID_SLOT;
  resultQueue->push(result);
}

void BlazeMemoryManager::requestEvict(uint32_t taskId, uint32_t slotId) {
  domain::ManageMemoryTask task;
  task.taskId = taskId;
  task.action = domain::MemoryManagementAction::EVICT;
  task.memoryLayout = domain::KvMemoryLayout::PAGED;
  task.slotId = slotId;
  handleRequest(task);
}

}  // namespace tt::services
