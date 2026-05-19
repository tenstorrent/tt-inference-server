#include "services/memory_services/blaze_memory_manager.hpp"

#include "tt_llm_engine/scheduler/decode/decode_types.hpp"

namespace tt::services {
namespace ds = tt_llm_engine::scheduler::decode;

std::optional<domain::ManageMemoryTask> BlazeMemoryManager::getRequest() {
  return MemoryManager::getRequest();
}

void BlazeMemoryManager::replyAllocateSuccess(uint32_t taskId,
                                              uint32_t slotId) {
  domain::ManageMemoryResult result{};
  result.taskId = taskId;
  result.status = domain::ManageMemoryStatus::SUCCESS;
  result.slotId = slotId;
  resultQueue->push(result);
}

void BlazeMemoryManager::replyAllocateFailure(uint32_t taskId) {
  domain::ManageMemoryResult result{};
  result.taskId = taskId;
  result.status = domain::ManageMemoryStatus::FAILURE;
  result.slotId = ds::INVALID_SLOT;
  resultQueue->push(result);
}

}  // namespace tt::services
