#include "services/memory_services/blaze_memory_manager.hpp"
#include "blaze_runner/blaze_utils.hpp"

#include "utils/logger.hpp"

namespace tt::services {
namespace utils = tt::runners::blaze_utils;
namespace ds = tt_llm_engine::scheduler::decode;

BlazeMemoryManager::BlazeMemoryManager(
    tt_llm_engine::scheduler::decode::DecodeScheduler& decodeScheduler,
    onEvictCb onEvict)
    : decodeScheduler(decodeScheduler), onEvict(onEvict) {}

std::optional<domain::ManageMemoryTask> BlazeMemoryManager::getRequest() {
  if (pendingRetry.has_value()) {
    auto task = std::move(*pendingRetry);
    pendingRetry.reset();
    return task;
  }
  return MemoryManager::getRequest();
}

void BlazeMemoryManager::handleRequest(
    const domain::ManageMemoryTask& request) {
  switch (request.action) {
    case domain::MemoryManagementAction::ALLOCATE: {
      if (!decodeScheduler.push_request(
              utils::makeAllocateRequest(request.taskId))) {
        TT_LOG_DEBUG(
            "[BlazeMemoryManager] ALLOCATE push_request failed; deferring "
            "retry for taskId={}",
            request.taskId);
        pendingRetry = request;
        return;
      }
      allocating.insert(request.taskId);
      TT_LOG_DEBUG(
          "[BlazeMemoryManager] ALLOCATE: taskId={},"
          "pending allocations={}",
          request.taskId, allocating.size());
      break;
    }
    case domain::MemoryManagementAction::DEALLOCATE: {
      auto slotId = request.slotId;
      if (!decodeScheduler.push_request(
              utils::makeCancelRequest(request.taskId, slotId))) {
        TT_LOG_DEBUG(
            "[BlazeMemoryManager] DEALLOCATE push_request failed for "
            "slotId={}; deferring retry for taskId={}",
            slotId, request.taskId);
        pendingRetry = request;
        return;
      }
      cancelling.insert({request.taskId, slotId});
      TT_LOG_DEBUG(
          "[BlazeMemoryManager] DEALLOCATE: taskId={}, slotId={}, "
          "pending cancellations={}",
          request.taskId, slotId, cancelling.size());
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
    if (slotId == ds::INVALID_SLOT) {
      TT_LOG_DEBUG(
          "[BlazeMemoryManager] handleResponse[ALLOCATE]: FAILURE "
          "No slot available for taskId={}",
          taskId);
      result.status = domain::ManageMemoryStatus::FAILURE;
      result.slotId = ds::INVALID_SLOT;
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
  if (auto it = cancelling.find(taskId); it != cancelling.end()) {
    auto recordedSlotId = it->second;
    cancelling.erase(it);
    if (slotId != recordedSlotId) {
      TT_LOG_ERROR(
          "[BlazeMemoryManager] handleResponse[CANCEL]: taskId={} "
          "ack slotId={} does not match recorded slotId={}; evicting "
          "the recorded slot",
          taskId, slotId, recordedSlotId);
    }
    TT_LOG_DEBUG(
        "[BlazeMemoryManager] handleResponse[CANCEL]: taskId={}, "
        "slotId={}, remaining pending cancellations={}",
        taskId, recordedSlotId, cancelling.size());
    onEvict(recordedSlotId);
    return;
  }
  TT_LOG_WARN(
      "[BlazeMemoryManager] handleResponse: unknown taskId={}, slotId={}",
      taskId, slotId);
}

}  // namespace tt::services
