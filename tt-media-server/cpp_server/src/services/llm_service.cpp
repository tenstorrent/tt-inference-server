// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/llm_service.hpp"

#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "config/settings.hpp"
#include "metrics/metrics.hpp"
#include "profiling/tracy.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "utils/logger.hpp"
#include "utils/mapper.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::services {

LLMService::LLMService() {
  const size_t numWorkers = tt::config::numWorkers();
  auto qm =
      std::make_unique<tt::ipc::QueueManager>(static_cast<int>(numWorkers));
  auto tq = qm->taskQueue;

  init(std::move(tq), std::make_unique<tt::worker::WorkerManager>(numWorkers),
       std::move(qm), tt::config::maxQueueSize());
}

LLMService::LLMService(std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
                       std::unique_ptr<tt::worker::WorkerManager> workerManager,
                       std::unique_ptr<tt::ipc::QueueManager> queueManager,
                       size_t maxQueueSize) {
  init(std::move(taskQueue), std::move(workerManager), std::move(queueManager),
       maxQueueSize);
}

void LLMService::init(std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
                      std::unique_ptr<tt::worker::WorkerManager> workerManager,
                      std::unique_ptr<tt::ipc::QueueManager> queueManager,
                      size_t maxQueueSize) {
  if (!taskQueue) {
    throw std::invalid_argument("LLMService: taskQueue must not be null");
  }
  if (!workerManager) {
    throw std::invalid_argument("LLMService: workerManager must not be null");
  }
  if (!queueManager) {
    throw std::invalid_argument("LLMService: queueManager must not be null");
  }

  this->taskQueue = std::move(taskQueue);
  this->workerManager = std::move(workerManager);
  this->queueManager = std::move(queueManager);
  this->maxQueueSize = maxQueueSize;

  const auto& stopIds = tt::utils::tokenizers::staticInfo().stopTokenIds;
  stopTokenSet = std::unordered_set<int64_t>(stopIds.begin(), stopIds.end());

  TT_LOG_INFO("[LLMService] Initialized (workers={})",
              this->workerManager->numWorkers());
}

LLMService::~LLMService() { stop(); }

void LLMService::start() {
  ZoneScopedN("LLMService::start");
  if (running.exchange(true)) {
    return;
  }

  TT_LOG_INFO("[LLMService] Starting (workers={})",
              workerManager->numWorkers());

  workerManager->start();
  tracy_config::tracyStartupSchedulerParent();
  startConsumers();

  TRACY_PLOT("pending_tasks", static_cast<double>(pendingTasks.load()));
  TT_LOG_INFO("[LLMService] Service started");
}

size_t LLMService::currentQueueSize() const { return pendingTasks.load(); }

bool LLMService::isModelReady() const { return workerManager->isReady(); }

std::vector<tt::worker::WorkerInfo> LLMService::getWorkerInfo() const {
  return workerManager->getWorkerInfo();
}

void LLMService::preProcess(LLMRequest& request) const {
  enforceQueueCapacity();

  if (std::holds_alternative<std::string>(request.prompt)) {
    request.prompt = tt::utils::tokenizers::activeTokenizer().encode(
        std::get<std::string>(request.prompt));
  }
}

void LLMService::startConsumers() {
  size_t n = workerManager->numWorkers();
  consumerThreads.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    consumerThreads.emplace_back(&LLMService::consumerLoopForWorker, this, i);
  }
  TT_LOG_INFO("[LLMService] Started {} consumer threads", n);
}

void LLMService::stop() {
  ZoneScopedN("LLMService::stop");
  if (!running.exchange(false)) {
    return;
  }

  TT_LOG_INFO("[LLMService] Stopping...");

  for (auto& q : queueManager->resultQueues) {
    q->shutdown();
  }

  for (auto& thread : consumerThreads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  consumerThreads.clear();

  workerManager->stop();

  TT_LOG_INFO("[LLMService] Stopped");
  queueManager->clear();
}

namespace {

std::string decodeToken(
    std::unordered_map<
        uint32_t,
        std::unique_ptr<tt::utils::tokenizers::Tokenizer::StreamDecoder>>&
        decoders,
    uint32_t taskId, uint32_t tokenId, bool isFinal, bool skipSpecial) {
  auto& decoder = decoders[taskId];
  if (!decoder) {
    decoder = tt::utils::tokenizers::activeTokenizer().createStreamDecoder(
        skipSpecial);
  }
  std::string delta = decoder->step(static_cast<int>(tokenId));
  if (isFinal) delta += decoder->flush();
  return delta;
}

LLMStreamChunk buildStreamChunk(
    const ipc::SharedToken& token, const std::string& delta,
    const std::unordered_set<int64_t>& stopTokenSet) {
  LLMStreamChunk response{token.task_id};
  response.id = std::to_string(token.task_id);
  response.created = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  LLMChoice choice;
  choice.index = token.token_index;
  choice.text = delta;

  choice.token_id = static_cast<int64_t>(token.token_id);
  choice.spec_accepts = token.spec_accepts;
  choice.spec_rejects = token.spec_rejects;
  if (token.isFinal()) {
    bool isStop = stopTokenSet.count(static_cast<int64_t>(token.token_id)) > 0;
    choice.finish_reason = isStop ? "stop" : "length";
  }
  response.choices.push_back(std::move(choice));
  return response;
}
}  // namespace

std::optional<LLMService::StreamCallbackEntry> LLMService::resolveCallback(
    uint32_t taskId, bool isFinal) {
  if (isFinal) {
    auto val = streamCallbacks.take(taskId);
    if (!val.has_value()) return std::nullopt;
    pendingTasks.fetch_sub(1);
    tt::metrics::ServerMetrics::instance().setQueueDepth(
        static_cast<double>(pendingTasks.load()));
    return std::move(val.value());
  }
  return streamCallbacks.get(taskId);
}

void LLMService::consumerLoopForWorker(size_t workerIdx) {
  ZoneScopedN("LLMService::consumer_loop");
  tracy_config::tracySetThreadName(
      ("Consumer-" + std::to_string(workerIdx)).c_str());

  TT_LOG_INFO("[Consumer-{}] Started", workerIdx);

  auto* worker = workerManager->worker(workerIdx);
  if (!worker || !worker->cfg.result_queue) {
    TT_LOG_WARN("[Consumer-{}] No token buffer, exiting", workerIdx);
    return;
  }
  auto resultQueue = worker->cfg.result_queue;

  std::unordered_map<
      uint32_t,
      std::unique_ptr<tt::utils::tokenizers::Tokenizer::StreamDecoder>>
      streamDecoders;

  while (running) {
    bool anyActivity = false;

    ipc::SharedToken token;
    while (resultQueue->blockingPop(token)) {
      anyActivity = true;

      uint32_t taskId = token.task_id;
      bool isError = token.isError();
      bool isFinal = token.isFinal();
      bool isAbort = token.isAbort();
      const auto errorReason = tt::ipc::errorReasonFromToken(token);

      if (isAbort && !isFinal) {
        // Client-initiated abort: LLMService::abortRequest has already taken
        // the callback, set the controller's done=true, and emitted the final
        // abort chunk. Just drop our local decoder state and move on.
        streamDecoders.erase(taskId);
        continue;
      }

      auto entry = resolveCallback(taskId, isFinal);
      if (!entry.has_value()) {
        streamDecoders.erase(taskId);
        continue;
      }
      if (isAbort) {
        // Runner-initiated preemption (isAbort && isFinal): the request was
        // dropped server-side (e.g. EVICT superseded an in-flight SUBMIT).
        // The client never aborted, so we must close the SSE stream ourselves
        // with a final abort chunk.
        streamDecoders.erase(taskId);
        tt::metrics::ServerMetrics::instance().onRequestCompleted(taskId,
                                                                  "abort");
        auto abortChunk = makeAbortChunk(taskId);
        entry->callback(abortChunk, /*isFinal=*/true);
        continue;
      }
      if (isError) {
        if (!isFinal) {
          // resolveCallback only takes/decrements when isFinal; mirror that
          // cleanup here so error tokens always terminate the stream cleanly.
          streamCallbacks.erase(taskId);
          pendingTasks.fetch_sub(1);
          tt::metrics::ServerMetrics::instance().setQueueDepth(
              static_cast<double>(pendingTasks.load()));
        }
        streamDecoders.erase(taskId);
        tt::metrics::ServerMetrics::instance().onRequestCompleted(
            taskId, finishReasonForError(errorReason));
        auto errorChunk = makeErrorChunk(taskId,
                                         errorReason == LLMErrorReason::TIMEOUT
                                             ? "runner timeout"
                                             : "runner reported error",
                                         errorReason);
        entry->callback(errorChunk, /*isFinal=*/true);
        continue;
      }

      // Dynamo path: skip decode (raw token_ids only).
      if (entry->skip_text_decode) {
        tt::metrics::ServerMetrics::instance().onToken(taskId);
        auto response = buildStreamChunk(token, /*delta=*/"", stopTokenSet);
        entry->callback(response, isFinal);
        if (isFinal) {
          std::optional<std::string> finalReason;
          if (!response.choices.empty() &&
              response.choices[0].finish_reason.has_value()) {
            finalReason = response.choices[0].finish_reason.value();
          }
          tt::metrics::ServerMetrics::instance().onRequestCompleted(
              taskId, finalReason.value_or("error"));
          streamDecoders.erase(taskId);
        }
        continue;
      }

      std::string delta = decodeToken(streamDecoders, taskId, token.token_id,
                                      isFinal, entry->skip_special_tokens);
      tt::metrics::ServerMetrics::instance().onToken(taskId);

  

      
        // Regular content.
        // Always emit chunks with token_id for Session hash tracking, even if
        // decoded text is empty. The controller callback uses token_id to
        // accumulate hashes; skipping here breaks prefix cache.
        auto response = buildStreamChunk(token, delta, stopTokenSet);
        entry->callback(response, isFinal);
        if (isFinal) {
          captureFinalFinishReason(response);
        }
      

      // Cleanup at finalization
      if (isFinal) {
        if (!finalFinishReason.has_value()) {
          TT_LOG_WARN(
              "[Consumer-{}] Final token for task {} reached cleanup without "
              "a finish reason set; defaulting to \"error\"",
              workerIdx, taskId);
        }
        tt::metrics::ServerMetrics::instance().onRequestCompleted(
            taskId, finalFinishReason.value_or("error"));
        streamDecoders.erase(taskId);



        TRACY_PLOT("pending_tasks", static_cast<double>(pendingTasks.load()));
      }
    }

    if (!anyActivity) {
      std::this_thread::yield();
    }
  }

  TT_LOG_INFO("[Consumer-{}] Stopped", workerIdx);
}

void LLMService::produceStream(
    LLMRequest request,
    std::function<void(LLMStreamChunk&, bool isFinal)> callback) {
  if (!callback) {
    throw std::invalid_argument("streaming callback must not be null");
  }

  ZoneScopedN("LLMService::produceStream");
  if (request.task_id == 0) {
    throw std::runtime_error("task_id must be set before submitting request");
  }
  uint32_t taskId = request.task_id;

  pendingTasks.fetch_add(1);
  TRACY_PLOT("pending_tasks", static_cast<double>(pendingTasks.load()));
  tt::metrics::ServerMetrics::instance().setQueueDepth(
      static_cast<double>(pendingTasks.load()));

  StreamCallbackEntry entry{std::move(callback), request.skip_special_tokens,
                            request.skip_text_decode};
  streamCallbacks.insert(taskId, std::move(entry));

  auto prompt = std::get<std::vector<int>>(request.prompt);
  std::vector<int64_t> tokenIds(prompt.begin(), prompt.end());

  tt::metrics::ServerMetrics::instance().onRequestSubmitted(
      taskId, static_cast<int>(tokenIds.size()));

  auto sequence = std::make_unique<tt::domain::llm::Sequence>(
      taskId,
      static_cast<int>(tt::config::llmEngineConfig().kvcache_block_size),
      std::move(tokenIds), prompt.size(), request.slotId, request.prefillSlotId,
      request.continuation, request.disaggregated,
      std::make_unique<tt::domain::llm::SamplingParams>(
          tt::utils::mapper::mapSamplingParams(request)),
      request.kv_position_id, request.decode_position_id,
      request.decode_skip_tokens, request.migrationId);
  taskQueue->push(*std::move(sequence));
}

void LLMService::abortRequest(uint32_t taskId) {
  // Atomically remove the stream callback and decrement pendingTasks.
  auto entry = streamCallbacks.take(taskId);
  if (entry.has_value()) {
    pendingTasks.fetch_sub(1);
    tt::metrics::ServerMetrics::instance().setQueueDepth(
        static_cast<double>(pendingTasks.load()));
  }

  tt::metrics::ServerMetrics::instance().onRequestCompleted(taskId, "abort");

  // Invoke the detached callback with isFinal=true. For streaming requests the
  // controller sets done=true BEFORE calling abortRequest, so the callback's
  // done->load() check returns immediately — no SSE data is sent.
  if (entry.has_value()) {
    auto abortChunk = makeAbortChunk(taskId);
    entry->callback(abortChunk, /*isFinal=*/true);
  }

  for (auto& cq : queueManager->cancelQueues) {
    cq->push(taskId);
  }

  TT_LOG_INFO("[LLMService] Aborted request {}", taskId);
}

}  // namespace tt::services
