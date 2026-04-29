// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/llm_service.hpp"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "config/settings.hpp"
#include "metrics/metrics.hpp"
#include "profiling/tracy.hpp"
#include "utils/logger.hpp"
#include "utils/mapper.hpp"
#include "utils/tokenizers/tokenizer.hpp"
#include "worker/worker_manager.hpp"

namespace tt::services {

// Bring ContentType and TokenParseResult into scope
using tt::services::ContentType;
using tt::services::TokenParseResult;

LLMService::LLMService()
    : tokenizer(&tt::utils::tokenizers::activeTokenizer()) {
  size_t numWorkers = tt::config::numWorkers();
  this->maxQueueSize = tt::config::maxQueueSize();

  const auto stopIds = tokenizer->stopTokenIds();
  stopTokenSet = std::unordered_set<int64_t>(stopIds.begin(), stopIds.end());

  workerManager = std::make_unique<tt::worker::WorkerManager>(numWorkers);
  reasoningParser = std::make_unique<ReasoningParser>();
  toolCallParser = createToolCallParser(tt::config::modelType());

  TT_LOG_INFO("[LLMService] Initialized (workers={})", numWorkers);
  queueManager =
      std::make_unique<tt::ipc::QueueManager>(static_cast<int>(numWorkers));
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

void LLMService::preProcess(domain::LLMRequest& request) const {
  BaseService::preProcess(request);

  if (request.tool_choice.has_value()) {
    const auto& type = request.tool_choice->type;
    if (type != "auto" && type != "none") {
      throw std::invalid_argument(
          "tool_choice='" + type +
          "' is not yet supported by this server; only 'auto' and 'none' are "
          "currently implemented");
    }
  }

  if (std::holds_alternative<std::string>(request.prompt)) {
    auto text = std::get<std::string>(request.prompt);
    static auto cfg = tt::utils::tokenizers::getTokenizerConfig();
    bool hasBos = text.size() >= cfg.bos_token.size() &&
                  text.compare(0, cfg.bos_token.size(), cfg.bos_token) == 0;
    if (cfg.add_bos_token && !cfg.bos_token.empty() && !hasBos) {
      text = cfg.bos_token + text;
    }
    request.prompt = tokenizer->encode(text);
  }
  const auto& tokens = std::get<std::vector<int>>(request.prompt);
  if (tokens.size() > tt::config::LLMConfig::MAX_INPUT_TOKENS) {
    throw std::invalid_argument(
        "Input too long: " + std::to_string(tokens.size()) +
        " tokens exceeds maximum of " +
        std::to_string(tt::config::LLMConfig::MAX_INPUT_TOKENS));
  }
  request.prompt_tokens_count = static_cast<int>(tokens.size());
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
    const tt::utils::tokenizers::Tokenizer* tokenizer, uint32_t taskId,
    uint32_t tokenId, bool isFinal, bool skipSpecial) {
  auto& decoder = decoders[taskId];
  if (!decoder) decoder = tokenizer->createStreamDecoder(skipSpecial);
  std::string delta = decoder->step(static_cast<int>(tokenId));
  if (isFinal) delta += decoder->flush();
  return delta;
}

domain::LLMStreamChunk buildStreamChunk(
    const ipc::SharedToken& token, const TokenParseResult& parseResult,
    const std::unordered_set<int64_t>& stopTokenSet) {
  domain::LLMStreamChunk response{token.task_id};
  response.id = std::to_string(token.task_id);
  response.created = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  domain::LLMChoice choice;
  choice.index = token.token_index;

  if (parseResult.type == ContentType::REASONING) {
    choice.text = "";
    choice.reasoning = parseResult.text;
  } else {
    choice.text = parseResult.text;
    choice.reasoning = std::nullopt;
  }

  if (token.isError()) {
    choice.finish_reason = "error";
  } else {
    choice.token_id = static_cast<int64_t>(token.token_id);
    if (token.isFinal()) {
      bool isStop =
          stopTokenSet.count(static_cast<int64_t>(token.token_id)) > 0;
      choice.finish_reason = isStop ? "stop" : "length";
    }
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
  if (!worker->cfg.result_queue) {
    TT_LOG_WARN("[Consumer-{}] No token buffer, exiting", workerIdx);
    return;
  }

  std::unordered_map<
      uint32_t,
      std::unique_ptr<tt::utils::tokenizers::Tokenizer::StreamDecoder>>
      streamDecoders;

  while (running) {
    bool anyActivity = false;

    ipc::SharedToken token;
    while (worker->cfg.result_queue->blockingPop(token)) {
      anyActivity = true;

      uint32_t taskId = token.task_id;
      bool isFinal = token.isFinal();

      auto entry = resolveCallback(taskId, isFinal);
      if (!entry.has_value()) {
        streamDecoders.erase(taskId);
        continue;
      }

      std::string delta =
          decodeToken(streamDecoders, tokenizer, taskId, token.token_id,
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

domain::LLMResponse LLMService::processRequest(domain::LLMRequest request) {
  ZoneScopedN("LLMService::processRequest");

  std::mutex mtx;
  std::condition_variable cv;
  bool done = false;

  std::string accumulatedAnswer;
  std::string accumulatedReasoning;
  int completionTokens = 0;
  std::string finishReason = "stop";

  const int promptTokens =
      std::holds_alternative<std::vector<int>>(request.prompt)
          ? static_cast<int>(std::get<std::vector<int>>(request.prompt).size())
          : 0;
  const uint32_t taskId = request.task_id;
  const std::string model = request.model.value_or("default");

  processStreamingRequest(
      std::move(request), [&](domain::LLMStreamChunk& chunk, bool isFinal) {
        if (!chunk.choices.empty()) {
          if (chunk.choices[0].reasoning.has_value()) {
            accumulatedReasoning.append(chunk.choices[0].reasoning.value());
          }
          accumulatedAnswer.append(chunk.choices[0].text);
          completionTokens++;
          if (chunk.choices[0].finish_reason.has_value()) {
            finishReason = chunk.choices[0].finish_reason.value();
          }
        }
        if (isFinal) {
          std::lock_guard<std::mutex> lock(mtx);
          done = true;
          cv.notify_one();
        }
      });

  std::unique_lock<std::mutex> lock(mtx);
  cv.wait(lock, [&] { return done; });

  domain::LLMResponse response{taskId};
  response.id = std::to_string(taskId);
  response.model = model;
  response.created = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  domain::LLMChoice choice;
  choice.text = std::move(accumulatedAnswer);
  choice.reasoning =
      accumulatedReasoning.empty()
          ? std::nullopt
          : std::optional<std::string>(std::move(accumulatedReasoning));
  choice.index = 0;
  choice.finish_reason = finishReason;
  response.choices.push_back(std::move(choice));

  response.usage = {
      promptTokens, completionTokens, promptTokens + completionTokens,
      std::nullopt, std::nullopt,     std::nullopt};

  return response;
}

void LLMService::processStreamingRequest(
    domain::LLMRequest request,
    std::function<void(domain::LLMStreamChunk&, bool isFinal)> callback) {
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
    toolChoiceMap.insert(taskId, request.tool_choice->type);
  }

  if (!request.enable_reasoning) {
    reasoningSuppressedMap.insert(taskId, true);
    if (reasoningParser) {
      request.stop_token_ids.push_back(
          static_cast<int>(ReasoningParser::THINK_START_TOKEN));
    }
  }

  if (reasoningParser) {
    reasoningParser->initializeTask(taskId);
  }

  auto prompt = std::get<std::vector<int>>(request.prompt);
  std::vector<int64_t> tokenIds(prompt.begin(), prompt.end());

  tt::metrics::ServerMetrics::instance().onRequestSubmitted(
      taskId, static_cast<int>(prompt.size()));

  auto sequence = std::make_unique<tt::domain::Sequence>(
      taskId,
      static_cast<int>(tt::config::llmEngineConfig().kvcache_block_size),
      std::move(tokenIds));
  sequence->setNumPromptTokens(prompt.size());
  if (request.slotId.has_value()) {
    sequence->setKVCacheSlot(request.slotId.value());
  }
  sequence->setContinuation(request.continuation);
  sequence->setDisaggregated(request.disaggregated);
  sequence->setSamplingParams(std::make_unique<tt::domain::SamplingParams>(
      tt::utils::mapper::mapSamplingParams(request)));
  queueManager->taskQueue->push(*std::move(sequence));
}

void LLMService::postProcess(domain::LLMResponse& response) const {
  // Parse and strip reasoning blocks from all choices
  if (reasoningParser) {
    for (auto& choice : response.choices) {
      auto result = reasoningParser->parseComplete(choice.text);

      // Replace text with answer only (reasoning stripped)
      choice.text = std::move(result.answer);
    }
  }

  auto toolChoiceOpt = toolChoiceMap.take(response.task_id);
  const bool toolCallsDisabled =
      toolChoiceOpt.has_value() && toolChoiceOpt.value() == "none";
  if (toolCallsDisabled) {
    TT_LOG_DEBUG("[LLMService] Skipping tool call parsing (tool_choice=none)");
  }

  if (toolCallParser) {
    for (auto& choice : response.choices) {
      if (toolCallsDisabled) {
        choice.text = toolCallParser->stripMarkers(choice.text);
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

  // Invoke the detached callback with isFinal=true so any blocking waiter
  // (e.g. processRequest's cv.wait) is unblocked.  For streaming requests the
  // controller sets done=true BEFORE calling abortRequest, so the callback's
  // done->load() check returns immediately — no SSE data is sent.
  if (entry.has_value()) {
    domain::LLMStreamChunk abortResponse{taskId};
    domain::LLMChoice choice;
    choice.finish_reason = "abort";
    abortResponse.choices.push_back(std::move(choice));
    entry->callback(abortResponse, /*isFinal=*/true);
  }

  reasoningSuppressedMap.take(taskId);

  if (reasoningParser) {
    reasoningParser->finalizeTask(taskId);
  }

  // Broadcast the cancel signal to every per-worker cancel queue.
  // Each worker's scheduler::abortRequest is idempotent for unknown task IDs.
  if (queueManager) {
    for (auto& cq : queueManager->cancelQueues) {
      cq->push(taskId);
    }
  }

  TT_LOG_INFO("[LLMService] Aborted request {}", taskId);
}

}  // namespace tt::services
