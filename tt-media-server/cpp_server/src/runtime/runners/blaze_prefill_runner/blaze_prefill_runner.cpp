// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/blaze_prefill_runner/blaze_prefill_runner.hpp"

#include "config/settings.hpp"
#include "ipc/helpers/token_push.hpp"
#include "services/memory_services/memory_manager.hpp"
#include "utils/logger.hpp"

namespace tt::runners {

static constexpr size_t MAX_MEMORY_DRAIN_PER_STEP = 8;

BlazePrefillRunner::BlazePrefillRunner(const config::LLMConfig& config,
                                       ipc::IResultQueue* resultQueue,
                                       tt::ipc::ITaskQueue* taskQueue)
    : config(config), resultQueue(resultQueue), taskQueue(taskQueue) {
  modelRunner = blaze_prefill::makeModelRunner(config);
  memoryManager = std::make_unique<tt::services::MemoryManager>();
  TT_LOG_INFO("[BlazePrefillRunner] MemoryManager created");
}

BlazePrefillRunner::~BlazePrefillRunner() {
  stop();
  if (modelRunner) {
    modelRunner->exit();
  }
}

void BlazePrefillRunner::run() {
  while (!stopped.load(std::memory_order_relaxed)) {
    // Service memory allocation requests from SessionManager
    drainMemoryRequests();

    // Get next sequence from task queue
    auto sequence = taskQueue->tryPop();
    if (!sequence) {
      std::this_thread::yield();
      continue;
    }
    TT_LOG_DEBUG("[BlazePrefillRunner] Starting prefill for task {}",
                 sequence->taskId);

    if (sequence->getNumberOfDecodeSkipTokens() > 0) {
      TT_LOG_INFO(
          "[BlazePrefillRunner] task {} has numberOfDecodeSkipTokens={}",
          sequence->taskId, sequence->getNumberOfDecodeSkipTokens());
    }

    auto result = modelRunner->forward(
        sequence->taskId, sequence->getTokenIds(), sequence->getKVCacheSlot());

    if (!result) {
      TT_LOG_DEBUG(
          "[BlazePrefillRunner] forward returned without result for task {}",
          sequence->taskId);
      break;  // stopped
    }

    TT_LOG_DEBUG("[BlazePrefillRunner] forward finished for task {}",
                 result->taskId);

    if (result->isError) {
      TT_LOG_WARN("[BlazePrefillRunner] Error token for task {}",
                  result->taskId);
      ipc::helpers::pushErrorToken(*resultQueue, result->taskId,
                                   result->isTimeoutError);
    } else {
      TT_LOG_DEBUG(
          "[BlazePrefillRunner] pushToken task_id={} token_id={} finished={}",
          result->taskId, result->tokenId, true);
      ipc::helpers::pushToken(*resultQueue, sequence->taskId, result->tokenId,
                              ipc::SharedToken::FLAG_FINAL);
    }

    // sequence automatically cleaned up at end of scope
  }
}

bool BlazePrefillRunner::warmup() {
  std::vector<int64_t> warmupTokens = {1};
  uint32_t warmupTaskId = 0;  // Use 0 for warmup task

  TT_LOG_DEBUG("[BlazePrefillRunner] warmup forward task_id={} token_count={}",
               warmupTaskId, warmupTokens.size());
  auto result = modelRunner->forward(warmupTaskId, warmupTokens,
                                     tt::domain::INVALID_SLOT_ID);
  if (!result || result->isError) {
    TT_LOG_ERROR("[BlazePrefillRunner] Warmup failed");
    return false;
  }

  TT_LOG_INFO("[BlazePrefillRunner] Warmup successful");
  return true;
}

void BlazePrefillRunner::stop() {
  TT_LOG_INFO("[BlazePrefillRunner] Stopping");
  stopped.store(true, std::memory_order_relaxed);
}

void BlazePrefillRunner::drainMemoryRequests() {
  for (size_t i = 0; i < MAX_MEMORY_DRAIN_PER_STEP; ++i) {
    auto request = memoryManager->getRequest();
    if (!request) return;

    if (request->action == domain::MemoryManagementAction::ALLOCATE) {
      uint32_t slotId = nextSlotId++;
      if (nextSlotId >= static_cast<uint32_t>(tt::config::dsMaxUsers())) {
        nextSlotId = 0;
      }
      TT_LOG_DEBUG(
          "[BlazePrefillRunner] drainMemoryRequests: ALLOCATE taskId={}, "
          "assigned slotId={}",
          request->taskId, slotId);
      memoryManager->replyAllocateSuccess(request->taskId, slotId);
    } else if (request->action == domain::MemoryManagementAction::DEALLOCATE) {
      TT_LOG_DEBUG(
          "[BlazePrefillRunner] drainMemoryRequests: DEALLOCATE taskId={}, "
          "slotId={} (no-op)",
          request->taskId, request->slotId);
      // No real deallocation needed in prefill-only mode
    }
  }
}

}  // namespace tt::runners
