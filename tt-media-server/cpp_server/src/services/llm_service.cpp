// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/llm_service.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "config/settings.hpp"
#include "metrics/metrics.hpp"
#include "profiling/tracy.hpp"
#include "utils/logger.hpp"
#include "utils/mapper.hpp"
#include "utils/tokenizers/tokenizer.hpp"
#include "utils/tool_call_id_generator.hpp"
#include "worker/worker_manager.hpp"

namespace tt::services {

// Bring ContentType and TokenParseResult into scope
using tt::services::ContentType;
using tt::services::TokenParseResult;

LLMService::LLMService() {
  const size_t numWorkers = tt::config::numWorkers();
  auto qm =
      std::make_unique<tt::ipc::QueueManager>(static_cast<int>(numWorkers));
  auto tq = qm->taskQueue;

  init(std::move(tq), std::make_unique<tt::worker::WorkerManager>(numWorkers),
       std::make_unique<ReasoningParser>(),
       createToolCallParser(tt::config::modelType()), std::move(qm),
       tt::config::maxQueueSize());
}

LLMService::LLMService(std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
                       std::unique_ptr<tt::worker::WorkerManager> workerManager,
                       std::unique_ptr<ReasoningParser> reasoningParser,
                       std::unique_ptr<IToolCallParser> toolCallParser,
                       std::unique_ptr<tt::ipc::QueueManager> queueManager,
                       size_t maxQueueSize) {
  init(std::move(taskQueue), std::move(workerManager),
       std::move(reasoningParser), std::move(toolCallParser),
       std::move(queueManager), maxQueueSize);
}

void LLMService::init(std::shared_ptr<tt::ipc::ITaskQueue> taskQueue,
                      std::unique_ptr<tt::worker::WorkerManager> workerManager,
                      std::unique_ptr<ReasoningParser> reasoningParser,
                      std::unique_ptr<IToolCallParser> toolCallParser,
                      std::unique_ptr<tt::ipc::QueueManager> queueManager,
                      size_t maxQueueSize) {
  if (!taskQueue) {
    throw std::invalid_argument("LLMService: taskQueue must not be null");
  }
  if (!workerManager) {
    throw std::invalid_argument("LLMService: workerManager must not be null");
  }

  this->taskQueue = std::move(taskQueue);
  this->workerManager = std::move(workerManager);
  this->reasoningParser = std::move(reasoningParser);
  this->toolCallParser = std::move(toolCallParser);
  this->queueManager = std::move(queueManager);
  this->maxQueueSize = maxQueueSize;

  const auto stopIds = tt::utils::tokenizers::activeTokenizer().stopTokenIds();
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

  TRACY_PLOT("pending_tasks", static_cast<double>(pending_tasks_.load()));
  TT_LOG_INFO("[LLMService] Service started");
}

size_t LLMService::currentQueueSize() const { return pendingTasks.load(); }

bool LLMService::isModelReady() const { return workerManager->isReady(); }

std::vector<tt::worker::WorkerInfo> LLMService::getWorkerInfo() const {
  return workerManager->getWorkerInfo();
}

void LLMService::preProcess(LLMRequest& request) const {
  BaseService::preProcess(request);

  if (std::holds_alternative<std::string>(request.prompt)) {
    auto text = std::get<std::string>(request.prompt);
    static auto cfg = tt::utils::tokenizers::getTokenizerConfig();
    bool hasBos = text.size() >= cfg.bos_token.size() &&
                  text.compare(0, cfg.bos_token.size(), cfg.bos_token) == 0;
    if (cfg.add_bos_token && !cfg.bos_token.empty() && !hasBos) {
      text = cfg.bos_token + text;
    }
    request.prompt = tt::utils::tokenizers::activeTokenizer().encode(text);
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

  if (queueManager) {
    for (auto& q : queueManager->resultQueues) {
      q->shutdown();
    }
  }

  for (auto& thread : consumerThreads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  consumerThreads.clear();

  workerManager->stop();

  TT_LOG_INFO("[LLMService] Stopped");
  if (queueManager) {
    queueManager->clear();
  }
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
    const ipc::SharedToken& token, const TokenParseResult& parseResult,
    const std::unordered_set<int64_t>& stopTokenSet) {
  LLMStreamChunk response{token.task_id};
  response.id = std::to_string(token.task_id);
  response.created = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  LLMChoice choice;
  choice.index = token.token_index;

  if (parseResult.type == ContentType::REASONING) {
    choice.text = "";
    choice.reasoning = parseResult.text;
  } else {
    choice.text = parseResult.text;
    choice.reasoning = std::nullopt;
  }

  choice.token_id = static_cast<int64_t>(token.token_id);
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

      auto entry = resolveCallback(taskId, isFinal);
      if (!entry.has_value()) {
        streamDecoders.erase(taskId);
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
        reasoningSuppressedMap.take(taskId);
        tt::metrics::ServerMetrics::instance().onRequestCompleted(taskId,
                                                                  "error");
        if (reasoningParser) {
          reasoningParser->finalizeTask(taskId);
        }
        auto errorChunk = makeErrorChunk(taskId, "runner reported error");
        entry->callback(errorChunk, /*isFinal=*/true);
        continue;
      }

      std::string delta = decodeToken(streamDecoders, taskId, token.token_id,
                                      isFinal, entry->skip_special_tokens);
      tt::metrics::ServerMetrics::instance().onToken(taskId);

      TokenParseResult parseResult{ContentType::ANSWER, delta, true};
      if (reasoningParser) {
        parseResult = reasoningParser->processToken(
            taskId, static_cast<int64_t>(token.token_id), delta);
      }

      bool suppressReasoning = reasoningSuppressedMap.get(taskId).has_value();
      if (suppressReasoning && parseResult.type == ContentType::REASONING) {
        if (isFinal) {
          parseResult = {ContentType::ANSWER, "", true};
        } else {
          continue;
        }
      }

      if ((!parseResult.should_emit || parseResult.text.empty()) && !isFinal) {
        continue;
      }

      auto response = buildStreamChunk(token, parseResult, stopTokenSet);
      entry->callback(response, isFinal);

      if (isFinal) {
        streamDecoders.erase(taskId);
        reasoningSuppressedMap.take(taskId);
        std::string finishReason = "unknown";
        if (!response.choices.empty() &&
            response.choices[0].finish_reason.has_value()) {
          finishReason = response.choices[0].finish_reason.value();
        }
        tt::metrics::ServerMetrics::instance().onRequestCompleted(taskId,
                                                                  finishReason);
        if (reasoningParser) {
          reasoningParser->finalizeTask(taskId);
        }
        TRACY_PLOT("pending_tasks", static_cast<double>(pendingTasks.load()));
      }
    }

    if (!anyActivity) {
      std::this_thread::yield();
    }
  }

  TT_LOG_INFO("[Consumer-{}] Stopped", workerIdx);
}

LLMResponse LLMService::processRequest(LLMRequest /*request*/) {
  throw std::runtime_error(
      "LLMService::processRequest is not supported; use streaming interface");
}

void LLMService::processStreamingRequest(
    LLMRequest request,
    std::function<void(LLMStreamChunk&, bool isFinal)> callback) {
  if (!callback) {
    throw std::invalid_argument("streaming callback must not be null");
  }

  ZoneScopedN("LLMService::processStreamingRequest");
  if (request.task_id == 0) {
    throw std::runtime_error("task_id must be set before submitting request");
  }
  uint32_t taskId = request.task_id;

  pendingTasks.fetch_add(1);
  TRACY_PLOT("pending_tasks", static_cast<double>(pendingTasks.load()));
  tt::metrics::ServerMetrics::instance().setQueueDepth(
      static_cast<double>(pendingTasks.load()));

  StreamCallbackEntry entry{std::move(callback), request.skip_special_tokens};
  streamCallbacks.insert(taskId, std::move(entry));

  if (request.tool_choice.has_value()) {
    toolChoiceMap.insert(taskId, request.tool_choice.value());
  }

  if (!request.enable_reasoning) {
    reasoningSuppressedMap.insert(taskId, true);
  }

  if (reasoningParser &&
      !request.disaggregated) {  // If request is disaggregated, dissagregation
                                 // service will initialize
    reasoningParser->initializeTask(taskId);
  }

  auto prompt = std::get<std::vector<int>>(request.prompt);
  std::vector<int64_t> tokenIds(prompt.begin(), prompt.end());

  tt::metrics::ServerMetrics::instance().onRequestSubmitted(
      taskId, static_cast<int>(prompt.size()));

  auto sequence = std::make_unique<tt::domain::llm::Sequence>(
      taskId,
      static_cast<int>(tt::config::llmEngineConfig().kvcache_block_size),
      std::move(tokenIds));
  sequence->setNumPromptTokens(prompt.size());
  if (request.slotId.has_value()) {
    sequence->setKVCacheSlot(request.slotId.value());
  }
  sequence->setContinuation(request.continuation);
  sequence->setDisaggregated(request.disaggregated);
  sequence->setSamplingParams(std::make_unique<tt::domain::llm::SamplingParams>(
      tt::utils::mapper::mapSamplingParams(request)));
  taskQueue->push(*std::move(sequence));
}

void LLMService::postProcess(LLMResponse& response) const {
  auto toolChoiceOpt = toolChoiceMap.take(response.task_id);
  tt::domain::tool_calls::ToolChoice toolChoice;
  if (toolChoiceOpt.has_value()) {
    toolChoice = std::move(toolChoiceOpt.value());
  } else {
    toolChoice.type = "auto";
  }

  if (!toolCallParser) {
    return;
  }

  for (size_t choiceIdx = 0; choiceIdx < response.choices.size(); ++choiceIdx) {
    auto& choice = response.choices[choiceIdx];
    if (toolChoice.type == "none") {
      choice.text = toolCallParser->stripMarkers(choice.text);
      continue;
    }

    if (toolChoice.type == "function") {
      // Generate unique tool call ID in OpenAI format (call_1, call_2, ...)
      std::string toolCallId = tt::utils::ToolCallIDGenerator::generate();

      Json::Value decodedOutput;
      static const Json::CharReaderBuilder kReaderBuilder;
      thread_local const std::unique_ptr<Json::CharReader> reader(
          kReaderBuilder.newCharReader());
      std::string argumentsStr = choice.text;
      std::string parseErrors;

      const bool parsed = reader->parse(choice.text.data(),
                                        choice.text.data() + choice.text.size(),
                                        &decodedOutput, &parseErrors);
      if (parsed && decodedOutput.isMember("arguments")) {
        argumentsStr = decodedOutput["arguments"].toStyledString();
      }

      Json::Value toolCallJson;
      toolCallJson["id"] = toolCallId;
      toolCallJson["type"] = "function";
      toolCallJson["function"]["name"] =
          toolChoice.function.value_or("unknown");
      toolCallJson["function"]["arguments"] = argumentsStr;

      Json::Value toolCallsArray(Json::arrayValue);
      toolCallsArray.append(toolCallJson);

      choice.tool_calls = toolCallsArray;
      choice.text = "";
      choice.finish_reason = "tool_calls";

      TT_LOG_DEBUG(
          "[LLMService] Created tool_call from structured output "
          "(tool_choice=function, function={}, id={})",
          toolChoice.function.value_or("unknown"), toolCallId);
      continue;
    }

    TT_LOG_DEBUG(
        "[LLMService] Parsing text for tool calls (length={}): {}",
        choice.text.length(),
        choice.text.substr(0, std::min<size_t>(200, choice.text.length())));

    auto toolCalls = toolCallParser->parseComplete(choice.text);
    if (toolCalls.has_value() && !toolCalls->empty()) {
      TT_LOG_DEBUG("[LLMService] Found {} tool calls", toolCalls->size());
      choice.tool_calls = std::move(toolCalls);
      choice.text = toolCallParser->stripMarkers(choice.text);
      choice.finish_reason = "tool_calls";
    }
  }
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
    LLMStreamChunk abortResponse{taskId};
    LLMChoice choice;
    choice.finish_reason = "abort";
    abortResponse.choices.push_back(std::move(choice));
    entry->callback(abortResponse, /*isFinal=*/true);
  }

  // Clean up any reasoning-parser state so task_states_ does not leak.
  reasoningSuppressedMap.take(taskId);

  if (reasoningParser) {
    reasoningParser->finalizeTask(taskId);
  }

  if (queueManager) {
    for (auto& cq : queueManager->cancelQueues) {
      cq->push(taskId);
    }
  }

  TT_LOG_INFO("[LLMService] Aborted request {}", taskId);
}

}  // namespace tt::services
