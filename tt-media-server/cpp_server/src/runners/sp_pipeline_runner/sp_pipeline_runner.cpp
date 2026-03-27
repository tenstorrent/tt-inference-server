// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <thread>

#include "profiling/tracy.hpp"
#include "utils/logger.hpp"
#include <pipeline_manager/pipeline_manager_types.hpp>


namespace tt::runners {

SpPipelineRunner::SpPipelineRunner(const config::LLMConfig& config,
                                   ipc::TokenRingBuffer<65536>* resultQueue,
                                   llm_engine::ITaskQueue* taskQueue)
    : config(config),
      stopTokenIds(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      resultQueue(resultQueue),
      taskQueue(taskQueue),
      maxInFlightCount(config.max_in_flight_count * 30) {
  memoryThread = std::thread([this] { memoryLoop(); });
  pipeline = std::make_unique<tt_blaze::pipeline_manager::MockPipeline>(); 
  pipelineManager = std::make_unique<tt_blaze::pipeline_manager::PipelineManager>(*pipeline);
  pipelineManager->start();
}

SpPipelineRunner::~SpPipelineRunner() {
  stop();
  if (memoryThread.joinable()) {
    memoryThread.join();
  }
  if (pipelineManager) {
    pipelineManager->stop();
  }
}

void SpPipelineRunner::run() {
  while (!stopped.load(std::memory_order_relaxed)) {
    step();
  }
}

bool SpPipelineRunner::warmup() {
  // Create a warmup sequence with a single token
  llm_engine::SamplingParams warmupParams;
  warmupParams.max_tokens = 1;
  warmupParams.ignore_eos = true;

  std::vector<int64_t> warmupTokens = {1};  // Single token
  llm_engine::TaskID warmupTaskId("warmup_task");

  auto warmupSeq = std::make_unique<llm_engine::Sequence>(
      warmupTaskId,
      1,  // block_size (doesn't matter for warmup)
      warmupTokens, warmupParams);
  
  pipelineManager->push_request(tt_blaze::pipeline_manager::ISRequest{
    .type = tt_blaze::pipeline_manager::RequestType::ALLOCATE,
  });
  
  auto response = tt_blaze::pipeline_manager::PMResponse{};
  pipelineManager->tick();
  while (!pipelineManager->try_pop_response(response)) {
    pipelineManager->tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  
  auto slotId = response.slot_id;
  if (slotId == tt_blaze::pipeline_manager::INVALID_SLOT) {
    TT_LOG_ERROR("SpPipelineRunner: Warmup failed with error");
    return false;
  }

  pipelineManager->push_request(tt_blaze::pipeline_manager::ISRequest{
    .type = tt_blaze::pipeline_manager::RequestType::SUBMIT,
    .slot_id = slotId,
    .token_count = static_cast<uint32_t>(warmupSeq->tokenIds.size()),
    .tokens = {static_cast<uint32_t>(warmupSeq->tokenIds[0])},
    .max_new_tokens = 1,
    .temperature = 1.0f,
    .top_p = 1.0f,
    .top_k = -1,
  });
  // Wait for the response token (with timeout)
  const int MAX_ATTEMPTS = 1000;  // ~10 seconds with 10ms sleep
  int attempts = 0;
  bool receivedToken = false;
  auto output = tt_blaze::pipeline_manager::OutputMessage{};
  pipelineManager->tick();

  while (attempts < MAX_ATTEMPTS && !receivedToken) {
    receivedToken = pipelineManager->try_pop_output(output);
    if (receivedToken) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    attempts++;
  }

  if (!receivedToken) {
    TT_LOG_ERROR("SpPipelineRunner: Warmup timed out waiting for token");
    return false;
  }

  TT_LOG_INFO("SpPipelineRunner: Warmup successful");
  pipelineManager->push_request(tt_blaze::pipeline_manager::ISRequest{
    .type = tt_blaze::pipeline_manager::RequestType::CANCEL,
    .slot_id = slotId,
  });
  pipelineManager->tick();
  return true;
}

void SpPipelineRunner::stop() {
  stopped.store(true, std::memory_order_relaxed);
}

void SpPipelineRunner::memoryLoop() {
  tt::domain::ManageMemoryTask task{};
  std::vector<tt::domain::ManageMemoryTask> retryQueue;

  while (!stopped.load(std::memory_order_relaxed)) {
    if (!retryQueue.empty()) {
      auto result = memoryManager.handle_task(retryQueue.front());
      if (result.status != domain::ManageMemoryStatus::WAITING) {
        memoryResults.push(result);
        retryQueue.erase(retryQueue.begin());
      }
    } else if (memoryRequests.tryPop(task)) {
      auto result = memoryManager.handle_task(task);
      if (result.status == domain::ManageMemoryStatus::WAITING) {
        retryQueue.push_back(task);
      } else {
        memoryResults.push(result);
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void SpPipelineRunner::step() {
  pipelineManager->tick();
  auto response = getResponse();
  if (response.has_value()) {
    handleResponse(*response);
  }
  auto output = getOutput();
  if (output.has_value()) {
    handleOutput(*output);
  }
  auto request = getRequest();
  if (request) {
    handleRequest(std::move(request));
  }
}

void SpPipelineRunner::pushToken(const llm_engine::TaskID& taskId,
                                 uint64_t tokenId, bool finished) {
  ipc::SharedToken shared{};
  shared.token_index = 0;
  shared.flags = finished ? ipc::SharedToken::FLAG_FINAL : 0u;
  shared.token_id = tokenId;
  std::strncpy(shared.task_id, taskId.id.c_str(), sizeof(shared.task_id) - 1);
  shared.task_id[sizeof(shared.task_id) - 1] = '\0';
  resultQueue->push(shared);
}

void SpPipelineRunner::pushErrorToken(const llm_engine::TaskID& taskId) {
  ipc::SharedToken shared{};
  shared.token_index = 0;
  shared.flags = ipc::SharedToken::FLAG_FINAL | ipc::SharedToken::FLAG_ERROR;
  shared.token_id = 0;
  std::strncpy(shared.task_id, taskId.id.c_str(), sizeof(shared.task_id) - 1);
  shared.task_id[sizeof(shared.task_id) - 1] = '\0';
  resultQueue->push(shared);
}

std::optional<tt_blaze::pipeline_manager::PMResponse> SpPipelineRunner::getResponse() {
  tt_blaze::pipeline_manager::PMResponse response;
  if (pipelineManager->try_pop_response(response)) {
    return response;
  }
  return std::nullopt;
}

std::optional<tt_blaze::pipeline_manager::OutputMessage> SpPipelineRunner::getOutput() {
  tt_blaze::pipeline_manager::OutputMessage output;
  if (pipelineManager->try_pop_output(output)) {
    return output;
  }
  return std::nullopt;
}

std::unique_ptr<llm_engine::Sequence> SpPipelineRunner::getRequest() {
  auto requestRaw = taskQueue->tryPop();
  if (!requestRaw) return nullptr;
  return std::unique_ptr<llm_engine::Sequence>(requestRaw);
}

bool SpPipelineRunner::handleResponse(tt_blaze::pipeline_manager::PMResponse& response) {
}

bool SpPipelineRunner::handleOutput(tt_blaze::pipeline_manager::OutputMessage& output) {
}


bool SpPipelineRunner::handleRequest(std::unique_ptr<llm_engine::Sequence> request) {
}

}  // namespace tt::runners
