// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/llm_service.hpp"

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
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

  const auto stopIds = tokenizer_->stopTokenIds();
  const std::unordered_set<int64_t> stopTokenSet(stopIds.begin(),
                                                 stopIds.end());

  std::unordered_map<uint32_t,
                     std::unique_ptr<tt::utils::Tokenizer::StreamDecoder>>
      streamDecoders;

  // Per-request state used to throttle ITL metric events.
  //
  // Mitigation: observe ITL once every kItlSampleStride consecutive tokens.
  // The reported quantiles remain statistically representative; the absolute
  // ITL values are accurate because the caller pre-computes elapsed time
  // between the two most recent CONSECUTIVE tokens rather than relying on the
  // spacing between sampled events.
  static constexpr int kItlSampleStride = 5;

  struct MetricsSamplingState {
    int token_count = 0;
    std::chrono::steady_clock::time_point prev_token_time;
  };
  std::unordered_map<uint32_t, MetricsSamplingState> metricsSampling;

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

      // For final tokens, atomically take ownership of the callback so that
      // only one of {consumer, abortRequest} decrements pending_tasks_.
      std::function<void(domain::LLMStreamChunk&, bool)> callback;
      if (isFinal) {
        auto val = stream_callbacks_.take(taskId);
        if (!val.has_value()) {
          // abortRequest already took the callback and finalized the task.
          streamDecoders.erase(taskId);
          metricsSampling.erase(taskId);
          continue;
        }
        callback = std::move(val.value());
        pending_tasks_.fetch_sub(1);
        tt::metrics::ServerMetrics::instance().setQueueDepth(
            static_cast<double>(pending_tasks_.load()));
      } else {
        auto val = stream_callbacks_.get(taskId);
        if (!val.has_value()) {
          // Client disconnected or task was cancelled — discard token.
          // abortRequest() already finalized the reasoning parser state.
          streamDecoders.erase(taskId);
          metricsSampling.erase(taskId);
          continue;
        }
        callback = std::move(val.value());
      }

      auto& decoder = streamDecoders[taskId];
      if (!decoder) decoder = tokenizer_->createStreamDecoder();

      std::string delta = decoder->step(static_cast<int>(token.token_id));
      if (isFinal) delta += decoder->flush();

      // Record token-level metrics before reasoning filtering so all
      // generated tokens (including hidden reasoning tokens) are counted.
      //
      // Hot-path budget: only a map counter increment for non-sampled tokens.
      // steady_clock::now() and metric queue pushes happen only on the first
      // token (TTFT) and every kItlSampleStride-th subsequent token.
      // The ITL value is divided by kItlSampleStride to yield a per-token
      // estimate (assumes roughly uniform decode step latency).
      {
        auto& ms = metricsSampling[taskId];
        if (ms.token_count == 0) {
          tt::metrics::ServerMetrics::instance().onToken(taskId);
          ms.prev_token_time = std::chrono::steady_clock::now();
        } else if (ms.token_count % kItlSampleStride == 0) {
          auto now = std::chrono::steady_clock::now();
          double itl =
              std::chrono::duration<double>(now - ms.prev_token_time).count() /
              kItlSampleStride;
          tt::metrics::ServerMetrics::instance().onITLSample(taskId, itl);
          ms.prev_token_time = now;
        }
        ms.token_count++;
      }

      TokenParseResult parseResult{ContentType::ANSWER, delta, true};
      if (reasoning_parser_) {
        parseResult = reasoning_parser_->processToken(
            taskId, static_cast<int64_t>(token.token_id), delta);
      }

      if ((!parseResult.should_emit || parseResult.text.empty()) && !isFinal) {
        continue;
      }

      domain::LLMStreamChunk response{token.task_id};
      response.id = std::to_string(taskId);
      response.created =
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();

      domain::LLMChoice choice;
      choice.index = token.token_index;

      // Set text based on content type
      if (parseResult.type == ContentType::REASONING) {
        // This is reasoning content - put in reasoning field
        choice.text = "";  // Empty normal text
        choice.reasoning = parseResult.text;
      } else {
        // This is answer content - put in text field
        choice.text = parseResult.text;
        choice.reasoning = std::nullopt;
      }

      if (token.isError()) {
        choice.finish_reason = "error";
      } else {
        choice.token_id = static_cast<int64_t>(token.token_id);
        if (isFinal) {
          bool isStop =
              stopTokenSet.count(static_cast<int64_t>(token.token_id)) > 0;
          choice.finish_reason = isStop ? "stop" : "length";
        }
      }
      response.choices.push_back(std::move(choice));

      callback(response, isFinal);

      if (isFinal) {
        streamDecoders.erase(taskId);
        std::string finishReason = "unknown";
        if (!response.choices.empty() &&
            response.choices[0].finish_reason.has_value()) {
          finishReason = response.choices[0].finish_reason.value();
        }
        int genTokens = metricsSampling[taskId].token_count;
        metricsSampling.erase(taskId);
        tt::metrics::ServerMetrics::instance().onRequestCompleted(
            taskId, finishReason, genTokens);
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
  assert(callback != nullptr);

  ZoneScopedN("LLMService::processStreamingRequest");
  if (request.task_id == 0) {
    throw std::runtime_error("task_id must be set before submitting request");
  }
  uint32_t taskId = request.task_id;

  pending_tasks_.fetch_add(1);
  TRACY_PLOT("pending_tasks", static_cast<double>(pending_tasks_.load()));
  tt::metrics::ServerMetrics::instance().setQueueDepth(
      static_cast<double>(pending_tasks_.load()));

  stream_callbacks_.insert(taskId, std::move(callback));

  if (reasoning_parser_) {
    reasoning_parser_->initializeTask(taskId);
  }

  auto prompt = std::get<std::vector<int>>(request.prompt);
  std::vector<int64_t> tokenIds(prompt.begin(), prompt.end());

  tt::metrics::ServerMetrics::instance().onRequestSubmitted(
      taskId, static_cast<int>(prompt.size()));

  auto sequence = std::make_unique<llm_engine::Sequence>(
      taskId,
      static_cast<int>(tt::config::llmEngineConfig().kvcache_block_size),
      std::move(tokenIds));
  sequence->numPromptTokens = prompt.size();
  if (request.slotId.has_value()) {
    sequence->setKVCacheAddress(request.slotId.value());
  }
  sequence->samplingParams = std::make_unique<llm_engine::SamplingParams>(
      tt::utils::mapper::mapSamplingParams(request));
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
  auto cb = stream_callbacks_.take(taskId);
  if (cb.has_value()) {
    pending_tasks_.fetch_sub(1);
    tt::metrics::ServerMetrics::instance().setQueueDepth(
        static_cast<double>(pending_tasks_.load()));
  }

  // Invoke the detached callback with isFinal=true so any blocking waiter
  // (e.g. processRequest's cv.wait) is unblocked.  For streaming requests the
  // controller sets done=true BEFORE calling abortRequest, so the callback's
  // done->load() check returns immediately — no SSE data is sent.
  if (cb.has_value()) {
    domain::LLMStreamChunk abortResponse{taskId};
    domain::LLMChoice choice;
    choice.finish_reason = "abort";
    abortResponse.choices.push_back(std::move(choice));
    cb.value()(abortResponse, /*isFinal=*/true);
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
