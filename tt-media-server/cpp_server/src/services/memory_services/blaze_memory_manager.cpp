#include "services/memory_services/blaze_memory_manager.hpp"

#include "runners/sp_pipeline_runner/blaze_utils.hpp"
#include "utils/logger.hpp"

namespace tt::services {
namespace utils = tt::runners::blaze_utils;

BlazeMemoryManager::BlazeMemoryManager(
    tt_blaze::pipeline_manager::PipelineManager& pipelineManager,
    onEvictCb onEvict)
    : pipelineManager(pipelineManager), onEvict(onEvict) {}

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
      auto requestId = nextRequestID;
      if (!pipelineManager.push_request(
              utils::makeAllocateRequest(requestId))) {
        TT_LOG_DEBUG(
            "[BlazeMemoryManager] ALLOCATE push_request failed; deferring "
            "retry for taskId={}",
            request.taskId);
        pendingRetry = request;
        return;
      }
      ++nextRequestID;
      allocating[requestId] = request.taskId;
      TT_LOG_DEBUG(
          "[BlazeMemoryManager] ALLOCATE: taskId={}, assigned "
          "requestId={}, pending allocations={}",
          request.taskId, requestId, allocating.size());
      break;
    }
    case domain::MemoryManagementAction::DEALLOCATE: {
      // Per slot, push CANCEL with a unique requestId, record it in the
      // cancelling map, and DO NOT call onEvict. Eviction is deferred until
      // handleResponse sees the matching PM ack, at which point the slot is
      // guaranteed to have been fully torn down on the PM side.
      size_t pushed = 0;
      for (; pushed < request.slotIds.size(); ++pushed) {
        auto slotId = request.slotIds[pushed];
        auto requestId = nextRequestID;
        if (!pipelineManager.push_request(
                utils::makeCancelRequest(requestId, slotId))) {
          TT_LOG_DEBUG(
              "[BlazeMemoryManager] DEALLOCATE push_request failed for "
              "slotId={}; deferring retry for {} remaining slot(s) of "
              "taskId={}",
              slotId, request.slotIds.size() - pushed, request.taskId);
          break;
        }
        ++nextRequestID;
        cancelling[requestId] = slotId;
        TT_LOG_DEBUG(
            "[BlazeMemoryManager] DEALLOCATE: taskId={}, slotId={}, "
            "cancel requestId={}, pending cancellations={}",
            request.taskId, slotId, requestId, cancelling.size());
      }
      if (pushed < request.slotIds.size()) {
        domain::ManageMemoryTask retry = request;
        retry.slotIds.assign(request.slotIds.begin() + pushed,
                             request.slotIds.end());
        pendingRetry = std::move(retry);
      }
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

void BlazeMemoryManager::handleResponse(uint32_t requestId, uint32_t slotId) {
  if (auto it = allocating.find(requestId); it != allocating.end()) {
    auto taskId = it->second;
    allocating.erase(it);
    TT_LOG_DEBUG(
        "[BlazeMemoryManager] handleResponse[ALLOCATE]: requestId={}, "
        "taskId={}, slotId={}, remaining pending={}",
        requestId, taskId, slotId, allocating.size());
    domain::ManageMemoryResult result;
    result.taskId = taskId;
    if (slotId == tt_blaze::pipeline_manager::INVALID_SLOT) {
      TT_LOG_DEBUG(
          "[BlazeMemoryManager] handleResponse[ALLOCATE]: FAILURE "
          "(INVALID_SLOT) for taskId={}",
          taskId);
      result.status = domain::ManageMemoryStatus::FAILURE;
      result.slotIds = {tt_blaze::pipeline_manager::INVALID_SLOT};
      resultQueue->push(result);
      return;
    }
    TT_LOG_DEBUG(
        "[BlazeMemoryManager] handleResponse[ALLOCATE]: SUCCESS taskId={}, "
        "slotId={}",
        taskId, slotId);
    result.status = domain::ManageMemoryStatus::SUCCESS;
    result.slotIds = {slotId};
    resultQueue->push(result);
    return;
  }
  if (auto it = cancelling.find(requestId); it != cancelling.end()) {
    auto recordedSlotId = it->second;
    cancelling.erase(it);
    if (slotId != recordedSlotId) {
      TT_LOG_WARN(
          "[BlazeMemoryManager] handleResponse[CANCEL]: requestId={} "
          "ack slotId={} does not match recorded slotId={}; evicting "
          "the recorded slot",
          requestId, slotId, recordedSlotId);
    }
    TT_LOG_DEBUG(
        "[BlazeMemoryManager] handleResponse[CANCEL]: requestId={}, "
        "slotId={}, remaining pending cancellations={}",
        requestId, recordedSlotId, cancelling.size());
    onEvict(recordedSlotId);
    return;
  }
  TT_LOG_WARN(
      "[BlazeMemoryManager] handleResponse: unknown requestId={}, slotId={}",
      requestId, slotId);
}

}  // namespace tt::services
