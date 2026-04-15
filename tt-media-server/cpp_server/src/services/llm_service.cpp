// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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
#include "utils/tokenizer.hpp"
#include "worker/worker_manager.hpp"

namespace tt::services {

// Bring ContentType and TokenParseResult into scope
using tt::services::ContentType;
using tt::services::TokenParseResult;

LLMService::LLMService() : tokenizer_(&tt::utils::activeTokenizer()) {
  size_t numWorkers = tt::config::numWorkers();
  max_queue_size_ = tt::config::maxQueueSize();

  const auto stopIds = tokenizer_->stopTokenIds();
  stop_token_set_ = std::unordered_set<int64_t>(stopIds.begin(), stopIds.end());

  worker_manager_ = std::make_unique<tt::worker::WorkerManager>(numWorkers);
  reasoning_parser_ = std::make_unique<ReasoningParser>();

  TT_LOG_INFO("[LLMService] Initialized (workers={})", numWorkers);
  queue_manager_ =
      std::make_unique<tt::ipc::QueueManager>(static_cast<int>(numWorkers));
}

LLMService::~LLMService() { stop(); }

void LLMService::start() {
  ZoneScopedN("LLMService::start");
  if (running_.exchange(true)) {
    return;
  }

  TT_LOG_INFO("[LLMService] Starting (workers={})",
              worker_manager_->numWorkers());

  worker_manager_->start();
  tracy_config::tracyStartupSchedulerParent();
  startConsumers();

  TRACY_PLOT("pending_tasks", static_cast<double>(pending_tasks_.load()));
  TT_LOG_INFO("[LLMService] Service started");
}

size_t LLMService::currentQueueSize() const { return pending_tasks_.load(); }

bool LLMService::isModelReady() const { return worker_manager_->isReady(); }

std::vector<tt::worker::WorkerInfo> LLMService::getWorkerInfo() const {
  return worker_manager_->getWorkerInfo();
}

void LLMService::preProcess(domain::LLMRequest& request) const {
  BaseService::preProcess(request);

  if (std::holds_alternative<std::string>(request.prompt)) {
    auto text = std::get<std::string>(request.prompt);
    static auto cfg = tt::utils::getTokenizerConfig();
    bool hasBos = text.size() >= cfg.bos_token.size() &&
                  text.compare(0, cfg.bos_token.size(), cfg.bos_token) == 0;
    if (cfg.add_bos_token && !cfg.bos_token.empty() && !hasBos) {
      text = cfg.bos_token + text;
    }
    request.prompt = tokenizer_->encode(text);
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
  size_t n = worker_manager_->numWorkers();
  consumer_threads_.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    consumer_threads_.emplace_back(&LLMService::consumerLoopForWorker, this, i);
  }
  TT_LOG_INFO("[LLMService] Started {} consumer threads", n);
}

void LLMService::stop() {
  ZoneScopedN("LLMService::stop");
  if (!running_.exchange(false)) {
    return;
  }

  TT_LOG_INFO("[LLMService] Stopping...");

  for (auto& q : queue_manager_->result_queues) {
    q->shutdown();
  }

  for (auto& thread : consumer_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  consumer_threads_.clear();

  worker_manager_->stop();

  TT_LOG_INFO("[LLMService] Stopped");
  queue_manager_->clear();
}

namespace {

std::string decodeToken(
    std::unordered_map<uint32_t,
                       std::unique_ptr<tt::utils::Tokenizer::StreamDecoder>>&
        decoders,
    const tt::utils::Tokenizer* tokenizer, uint32_t taskId, uint32_t tokenId,
    bool isFinal, bool skipSpecial) {
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
    auto val = stream_callbacks_.take(taskId);
    if (!val.has_value()) return std::nullopt;
    pending_tasks_.fetch_sub(1);
    tt::metrics::ServerMetrics::instance().setQueueDepth(
        static_cast<double>(pending_tasks_.load()));
    return std::move(val.value());
  }
  return stream_callbacks_.get(taskId);
}

void LLMService::consumerLoopForWorker(size_t workerIdx) {
  ZoneScopedN("LLMService::consumer_loop");
  tracy_config::tracySetThreadName(
      ("Consumer-" + std::to_string(workerIdx)).c_str());

  TT_LOG_INFO("[Consumer-{}] Started", workerIdx);

  auto* worker = worker_manager_->worker(workerIdx);
  if (!worker->cfg.result_queue) {
    TT_LOG_WARN("[Consumer-{}] No token buffer, exiting", workerIdx);
    return;
  }

  std::unordered_map<uint32_t,
                     std::unique_ptr<tt::utils::Tokenizer::StreamDecoder>>
      streamDecoders;

  while (running_) {
    if (!worker_manager_->checkWorkerAlive(workerIdx)) {
      TT_LOG_ERROR("[Consumer-{}] Worker process died, exiting consumer",
                   workerIdx);
      break;
    }

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
          decodeToken(streamDecoders, tokenizer_, taskId, token.token_id,
                      isFinal, entry->skip_special_tokens);
      tt::metrics::ServerMetrics::instance().onToken(taskId);

      TokenParseResult parseResult{ContentType::ANSWER, delta, true};
      if (reasoning_parser_) {
        parseResult = reasoning_parser_->processToken(
            taskId, static_cast<int64_t>(token.token_id), delta);
      }

      if ((!parseResult.should_emit || parseResult.text.empty()) && !isFinal) {
        continue;
      }

      auto response = buildStreamChunk(token, parseResult, stop_token_set_);
      entry->callback(response, isFinal);

      if (isFinal) {
        streamDecoders.erase(taskId);
        std::string finishReason = "unknown";
        if (!response.choices.empty() &&
            response.choices[0].finish_reason.has_value()) {
          finishReason = response.choices[0].finish_reason.value();
        }
        tt::metrics::ServerMetrics::instance().onRequestCompleted(
            taskId, finishReason);
        if (reasoning_parser_) {
          reasoning_parser_->finalizeTask(taskId);
        }
        TRACY_PLOT("pending_tasks", static_cast<double>(pending_tasks_.load()));
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

  pending_tasks_.fetch_add(1);
  TRACY_PLOT("pending_tasks", static_cast<double>(pending_tasks_.load()));
  tt::metrics::ServerMetrics::instance().setQueueDepth(
      static_cast<double>(pending_tasks_.load()));

  StreamCallbackEntry entry{std::move(callback), request.skip_special_tokens};
  stream_callbacks_.insert(taskId, std::move(entry));

  if (reasoning_parser_) {
    reasoning_parser_->initializeTask(taskId);
  }

  auto prompt = std::get<std::vector<int>>(request.prompt);
  std::vector<int64_t> tokenIds(prompt.begin(), prompt.end());

  tt::metrics::ServerMetrics::instance().onRequestSubmitted(
      taskId, static_cast<int>(prompt.size()));

  auto sequence = std::make_unique<tt::runners::llm_engine::Sequence>(
      taskId,
      static_cast<int>(tt::config::llmEngineConfig().kvcache_block_size),
      std::move(tokenIds));
  sequence->setNumPromptTokens(prompt.size());
  if (request.slotId.has_value()) {
    sequence->setKVCacheSlot(request.slotId.value());
  }
  sequence->setContinuation(request.continuation);
  sequence->setSamplingParams(
      std::make_unique<tt::runners::llm_engine::SamplingParams>(
          tt::utils::mapper::mapSamplingParams(request)));
  queue_manager_->task_queue->push(*std::move(sequence));
}

void LLMService::postProcess(domain::LLMResponse& response) const {
  // Parse and strip reasoning blocks from all choices
  if (reasoning_parser_) {
    for (auto& choice : response.choices) {
      auto result = reasoning_parser_->parseComplete(choice.text);

      // Replace text with answer only (reasoning stripped)
      choice.text = std::move(result.answer);
    }
  }
}

void LLMService::abortRequest(uint32_t taskId) {
  // Atomically remove the stream callback and decrement pending_tasks_.
  auto entry = stream_callbacks_.take(taskId);
  if (entry.has_value()) {
    pending_tasks_.fetch_sub(1);
    tt::metrics::ServerMetrics::instance().setQueueDepth(
        static_cast<double>(pending_tasks_.load()));
  }

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

  // Clean up any reasoning-parser state so task_states_ does not leak.
  if (reasoning_parser_) {
    reasoning_parser_->finalizeTask(taskId);
  }

  // Broadcast the cancel signal to every per-worker cancel queue.
  // Each worker's scheduler::abortRequest is idempotent for unknown task IDs.
  if (queue_manager_) {
    for (auto& cq : queue_manager_->cancel_queues) {
      cq->push(taskId);
    }
  }

  TT_LOG_INFO("[LLMService] Aborted request {}", taskId);
}

}  // namespace tt::services
