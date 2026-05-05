#include "services/memory_services/blaze_memory_manager.hpp"

#include <iostream>

#include "runners/sp_pipeline_runner/blaze_utils.hpp"
#include "utils/logger.hpp"

namespace tt::services {
namespace utils = tt::runners::blaze_utils;

BlazeMemoryManager::BlazeMemoryManager(
    tt_blaze::pipeline_manager::PipelineManager& pipelineManager,
    onEvictCb onEvict)
    : pipelineManager(pipelineManager), onEvict(onEvict) {}

void BlazeMemoryManager::handleRequest(
    const domain::ManageMemoryTask& request) {
  switch (request.action) {
    case domain::MemoryManagementAction::ALLOCATE: {
      auto requestId = nextRequestID++;
      allocating[requestId] = request.taskId;
      std::cout << "[MM] ALLOCATE_REQ task=" << request.taskId
                << " request_id=" << requestId
                << " pending=" << allocating.size() << std::endl;
      TT_LOG_DEBUG(
          "[BlazeMemoryManager] ALLOCATE: taskId={}, assigned "
          "requestId={}, pending allocations={}",
          request.taskId, requestId, allocating.size());
      pipelineManager.push_request(utils::makeAllocateRequest(requestId));
      break;
    }
    case domain::MemoryManagementAction::DEALLOCATE: {
      for (auto slotId : request.slotIds) {
        std::cout << "[MM] CANCEL_AND_EVICT task=" << request.taskId
                  << " slot=" << slotId << std::endl;
        TT_LOG_DEBUG(
            "[BlazeMemoryManager] DEALLOCATE: taskId={}, cancelling "
            "slotId={}, then evicting",
            request.taskId, slotId);
        pipelineManager.push_request(utils::makeCancelRequest(slotId));
        onEvict(slotId);
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
  auto it = allocating.find(requestId);
  if (it == allocating.end()) {
    std::cout << "[MM] UNKNOWN_RESPONSE request_id=" << requestId
              << " slot=" << slotId << std::endl;
    TT_LOG_WARN(
        "[BlazeMemoryManager] handleResponse: unknown requestId={}, "
        "slotId={}",
        requestId, slotId);
    return;
  }
  auto taskId = it->second;
  allocating.erase(it);
  TT_LOG_DEBUG(
      "[BlazeMemoryManager] handleResponse: requestId={}, taskId={}, "
      "slotId={}, remaining pending={}",
      requestId, taskId, slotId, allocating.size());
  domain::ManageMemoryResult result;
  result.taskId = taskId;
  if (slotId == tt_blaze::pipeline_manager::INVALID_SLOT) {
    std::cout << "[MM] ALLOCATE_FAIL task=" << taskId << " (INVALID_SLOT)"
              << std::endl;
    TT_LOG_DEBUG(
        "[BlazeMemoryManager] handleResponse: FAILURE (INVALID_SLOT) "
        "for taskId={}",
        taskId);
    result.status = domain::ManageMemoryStatus::FAILURE;
    result.slotIds = {tt_blaze::pipeline_manager::INVALID_SLOT};
    resultQueue->push(result);
    return;
  }
  std::cout << "[MM] ALLOCATE_OK task=" << taskId << " slot=" << slotId
            << " request_id=" << requestId << std::endl;
  TT_LOG_DEBUG(
      "[BlazeMemoryManager] handleResponse: SUCCESS taskId={}, "
      "slotId={}",
      taskId, slotId);
  result.status = domain::ManageMemoryStatus::SUCCESS;
  result.slotIds = {slotId};
  resultQueue->push(result);
}

}  // namespace tt::services
