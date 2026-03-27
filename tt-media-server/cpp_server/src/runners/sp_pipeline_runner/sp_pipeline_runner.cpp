// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/sp_pipeline_runner/sp_pipeline_runner.hpp"

#include <array>
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

inline void fillSequenceFields(pm::ISRequest& req,
                               const llm_engine::Sequence& seq) {
  req.token_count = static_cast<uint32_t>(seq.tokenIds.size());
  for (uint32_t i = 0; i < req.token_count && i < pm::MAX_SEQ_LEN; i++) {
    req.tokens[i] = static_cast<uint32_t>(seq.tokenIds[i]);
  }
  req.max_new_tokens = seq.samplingParams->max_tokens.value_or(
      static_cast<int>(tt::config::LLMConfig::MAX_INPUT_TOKENS));
  req.temperature = seq.samplingParams->temperature;
  req.top_p = seq.samplingParams->top_p.value_or(1.0f);
  req.top_k = seq.samplingParams->top_k.value_or(-1);
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
      taskQueue(taskQueue),
      maxInFlightCount(config.max_in_flight_count * 30) {
  memoryThread = std::thread([this] { memoryLoop(); });
  pipeline = std::make_unique<tt_blaze::pipeline_manager::MockPipeline>();
  pipelineManager =
      std::make_unique<tt_blaze::pipeline_manager::PipelineManager>(*pipeline);
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

  pipelineManager->push_request(makeAllocateRequest(nextRequestID++));

  pm::PMResponse response{};
  pipelineManager->tick();
  while (!pipelineManager->try_pop_response(response)) {
    pipelineManager->tick();
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
  pipelineManager->push_request(makeCancelRequest(slotId));
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
  // an open question: do we want to drain the task, response and output queues
  // here? or just pop each one once every iteration? I am afraid of starvation.

  pipelineManager->tick();  // temporary until PM gains the API thread ?
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
  if (inFlightCount >= maxInFlightCount) return nullptr;
  auto requestRaw = taskQueue->tryPop();
  if (!requestRaw) return nullptr;
  return std::unique_ptr<llm_engine::Sequence>(requestRaw);
}

void SpPipelineRunner::handleResponse(pm::PMResponse& response) {
  auto it = requestToSequence.find(response.request_id);
  if (it == requestToSequence.end()) {
    TT_LOG_ERROR("SpPipelineRunner: Response for unknown request");
    return;
  }
  auto seq = std::move(it->second);
  requestToSequence.erase(it);
  if (response.error_code != 0) {
    TT_LOG_ERROR(
        "SpPipelineRunner: Response for request {} returned error code {}",
        response.request_id, response.error_code);
    pushErrorToken(seq->taskId);
    return;
  }
  pipelineManager->push_request(makeSubmitRequest(response.slot_id, *seq));
  seq->setKVCacheAddress(response.slot_id);
  slotToSequence[response.slot_id] = std::move(seq);
  ++inFlightCount;
}

void SpPipelineRunner::handleOutput(pm::OutputMessage& output) {
  auto it = slotToSequence.find(output.slot_id);
  if (it == slotToSequence.end()) {
    TT_LOG_ERROR("SpPipelineRunner: Output for unknown slot");
    return;
  }
  auto& seq = *it->second;
  seq.appendToken(output.token_id);
  bool finished = output.is_complete || stopTokenIds.count(output.token_id);
  pushToken(seq.taskId, output.token_id, finished);
  if (finished) {
    // We need a mechanism here for when to evict in the future, we dont support
    // multi-turn conversations yet because we cancel the slot after the first
    // prompt.
    pipelineManager->push_request(makeCancelRequest(output.slot_id));
    slotToSequence.erase(it);
    --inFlightCount;
  }
}

void SpPipelineRunner::handleRequest(
    std::unique_ptr<llm_engine::Sequence> request) {
  if (auto slotId = request->getKVCacheAddress();
      slotId != llm_engine::INVALID_KV_CACHE_ADDRESS) {
    auto slot = static_cast<uint32_t>(slotId);
    pipelineManager->push_request(makeContinueRequest(slot, *request));
    slotToSequence[slot] = std::move(request);
    return;
  }
  pipelineManager->push_request(makeAllocateRequest(nextRequestID));
  requestToSequence[nextRequestID++] = std::move(request);
}

}  // namespace tt::runners
