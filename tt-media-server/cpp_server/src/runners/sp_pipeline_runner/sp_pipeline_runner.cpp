// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <pipeline_manager/pipeline_manager_types.hpp>
#include <thread>

#include "llm_runner/sequence.hpp"
#include "utils/logger.hpp"

namespace {

namespace pm = tt_blaze::pipeline_manager;

inline pm::ISRequest makeAllocateRequest(uint32_t requestId) {
  return {.type = pm::RequestType::ALLOCATE, .request_id = requestId};
}

inline pm::ISRequest makeCancelRequest(uint32_t slotId) {
  return {.type = pm::RequestType::CANCEL, .slot_id = slotId};
}

inline pm::GenerationParams makeGenerationParams(
    const llm_engine::Sequence& seq) {
  return {
      .max_new_tokens =
          static_cast<uint32_t>(seq.samplingParams->max_tokens.value_or(
              static_cast<int>(tt::config::LLMConfig::MAX_INPUT_TOKENS))),
      .temperature = seq.samplingParams->temperature,
      .top_p = seq.samplingParams->top_p.value_or(1.0f),
      .top_k = static_cast<int32_t>(seq.samplingParams->top_k.value_or(-1))};
}

inline void fillSequenceFields(pm::ISRequest& req,
                               const llm_engine::Sequence& seq) {
  req.tokens.assign(seq.tokenIds.begin(), seq.tokenIds.end());
  req.gen = makeGenerationParams(seq);
}

inline pm::ISRequest makeSubmitRequest(uint32_t slotId,
                                       const llm_engine::Sequence& seq) {
  pm::ISRequest req{};
  req.type = pm::RequestType::SUBMIT;
  req.slot_id = slotId;
  fillSequenceFields(req, seq);
  return req;
}

inline pm::ISRequest makeContinueRequest(uint32_t slotId,
                                         const llm_engine::Sequence& seq) {
  pm::ISRequest req{};
  req.type = pm::RequestType::CONTINUE;
  req.slot_id = slotId;
  fillSequenceFields(req, seq);
  return req;
}

}  // namespace

namespace tt::runners {

SpPipelineRunner::SpPipelineRunner(const config::LLMConfig& config,
                                   ipc::TokenRingBuffer<65536>* resultQueue,
                                   llm_engine::ITaskQueue* taskQueue)
    : config(config),
      stopTokenIds(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      resultQueue(resultQueue),
      taskQueue(taskQueue) {
  pm::MockConfig mockConfig{};
  pipelineManager = std::make_unique<pm::PipelineManager>(mockConfig);
  pipelineManager->start();
}

SpPipelineRunner::~SpPipelineRunner() {
  stop();
  if (pipelineManager) {
    pipelineManager->stop();
  }
}

void SpPipelineRunner::run() {
  while (!stopped.load(std::memory_order_relaxed)) {
    try {
      step();
    } catch (const std::exception& e) {
      TT_LOG_ERROR("SpPipelineRunner: Exception in run: {}", e.what());
      throw;
    }
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

  pipelineManager->push_request(makeAllocateRequest(nextRequestID++));

  pm::PMResponse response{};
  while (!pipelineManager->try_pop_response(response)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  auto slotId = response.slot_id;
  if (slotId == pm::INVALID_SLOT) {
    TT_LOG_ERROR("SpPipelineRunner: Warmup failed with error");
    return false;
  }

  pipelineManager->push_request(makeSubmitRequest(slotId, *warmupSeq));
  // Wait for the response token (with timeout)
  const int MAX_ATTEMPTS = 1000;  // ~10 seconds with 10ms sleep
  int attempts = 0;
  bool receivedToken = false;
  auto output = pm::OutputMessage{};

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
  pipelineManager->push_request(makeCancelRequest(slotId));
  return true;
}

void SpPipelineRunner::stop() {
  stopped.store(true, std::memory_order_relaxed);
}

void SpPipelineRunner::step() {
  // an open question: do we want to drain the task, response and output queues
  // here? or just pop each one once every iteration? I am afraid of starvation.

  auto memoryRequest = getMemoryRequest();
  if (memoryRequest.has_value()) {
    handleMemoryRequest(*memoryRequest);
  }
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

std::optional<pm::PMResponse> SpPipelineRunner::getResponse() {
  pm::PMResponse response;
  if (pipelineManager->try_pop_response(response)) {
    return response;
  }
  return std::nullopt;
}

std::optional<pm::OutputMessage> SpPipelineRunner::getOutput() {
  pm::OutputMessage output;
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

std::optional<tt::domain::ManageMemoryTask> SpPipelineRunner::getMemoryRequest() {
  tt::domain::ManageMemoryTask request;
  if (memoryRequests.tryPop(request)) {
    return request;
  }
  return std::nullopt;
}

void SpPipelineRunner::handleMemoryRequest(const tt::domain::ManageMemoryTask& request) {
  switch (request.action) {
    case tt::domain::MemoryManagementAction::ALLOCATE: {
      throw std::runtime_error("SpPipelineRunner: Allocate memory request not implemented");
    }
    case tt::domain::MemoryManagementAction::DEALLOCATE: {
      evictSlot(request.slotId);
      break;
    }
    case tt::domain::MemoryManagementAction::MOVE: {
      throw std::runtime_error("SpPipelineRunner: Move memory action not implemented");
    }
  }
}

void SpPipelineRunner::handleResponse(const pm::PMResponse& response) {
  auto it = allocating.find(response.request_id);
  if (it == allocating.end()) {
    TT_LOG_ERROR("SpPipelineRunner: Response for unknown request");
    return;
  }
  auto seq = std::move(it->second);
  allocating.erase(it);
  if (response.error_code != 0) {
    TT_LOG_ERROR(
        "SpPipelineRunner: Response for request {} returned error code {}",
        response.request_id, response.error_code);
    pushErrorToken(seq->taskId);
    return;
  }
  pipelineManager->push_request(makeSubmitRequest(response.slot_id, *seq));
  seq->setKVCacheAddress(response.slot_id);
  running[response.slot_id] = std::move(seq);
}

void SpPipelineRunner::handleOutput(const pm::OutputMessage& output) {
  auto it = running.find(output.slot_id);
  if (it == running.end()) {
    TT_LOG_ERROR("SpPipelineRunner: Output for unknown slot");
    return;
  }
  auto& seq = *it->second;
  seq.appendToken(output.token_id);
  bool finished = output.is_complete || stopTokenIds.count(output.token_id);
  pushToken(seq.taskId, output.token_id, finished);
}

void SpPipelineRunner::evictSlot(uint32_t slotId) {
  pipelineManager->push_request(makeCancelRequest(slotId));
  running.erase(slotId);
}

void SpPipelineRunner::handleRequest(
    std::unique_ptr<llm_engine::Sequence> request) {
  if (auto slotId = request->getKVCacheAddress();
      slotId != llm_engine::INVALID_KV_CACHE_ADDRESS) {
    auto slot = static_cast<uint32_t>(slotId);
    pipelineManager->push_request(makeContinueRequest(slot, *request));
    running[slot] = std::move(request);
    return;
  }
  pipelineManager->push_request(makeAllocateRequest(nextRequestID));
  allocating[nextRequestID++] = std::move(request);
}

}  // namespace tt::runners
