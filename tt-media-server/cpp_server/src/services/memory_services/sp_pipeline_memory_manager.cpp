#include "services/memory_services/sp_pipeline_memory_manager.hpp"

#include "runners/sp_pipeline_runner/sp_pipeline_utils.hpp"

namespace tt::services {
namespace utils = tt::runners::sp_pipeline_utils;

SpPipelineMemoryManager::SpPipelineMemoryManager(
    tt_blaze::pipeline_manager::PipelineManager& pipelineManager, onEvictCb onEvict)
    : pipelineManager(pipelineManager), onEvict(onEvict) {}

void SpPipelineMemoryManager::handleRequest(
    const domain::ManageMemoryTask& request) {
  switch (request.action) {
    case domain::MemoryManagementAction::ALLOCATE: {
      auto requestId = nextRequestID++;
      allocating[requestId] = request.taskId;
      pipelineManager.push_request(utils::makeAllocateRequest(requestId));
      break;
    }
    case domain::MemoryManagementAction::DEALLOCATE: {
      for (auto slotId : request.slotIds) {
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

void SpPipelineMemoryManager::handleResponse(uint32_t requestId,
                                             uint32_t slotId) {
  auto taskId = allocating[requestId];
  allocating.erase(requestId);
  domain::ManageMemoryResult result;
  result.taskId = taskId;
  if (slotId == tt_blaze::pipeline_manager::INVALID_SLOT) {
    result.status = domain::ManageMemoryStatus::FAILURE;
    result.slotIds = {tt_blaze::pipeline_manager::INVALID_SLOT};
    resultQueue->push(result);
    return;
  }
  result.status = domain::ManageMemoryStatus::SUCCESS;
  result.slotIds = {slotId};
  resultQueue->push(result);
}

}  // namespace tt::services
