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

void LLMService::preProcess(domain::CompletionRequest& request) const {
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

  const auto STOP_IDS = tokenizer_->stopTokenIds();
  const std::unordered_set<int64_t> STOP_TOKEN_SET(STOP_IDS.begin(),
                                                   STOP_IDS.end());

  std::unordered_map<std::string,
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

      std::string taskId = std::string(token.task_id);
      auto val = stream_callbacks_.get(token.task_id);
      if (!val.has_value()) {
        // Client disconnected or task was cancelled — discard remaining
        // tokens and clean up local state for this task.
        streamDecoders.erase(taskId);
        if (reasoning_parser_) {
          reasoning_parser_->finalizeTask(taskId);
        }
        continue;
      }
      auto callback = val.value();

      bool isFinal = token.isFinal();

      if (isFinal) {
        stream_callbacks_.erase(token.task_id);
        pending_tasks_.fetch_sub(1);
        tt::metrics::ServerMetrics::instance().setQueueDepth(
            static_cast<double>(pending_tasks_.load()));
      }

      auto& decoder = streamDecoders[taskId];
      if (!decoder) decoder = tokenizer_->createStreamDecoder();

      std::string delta = decoder->step(static_cast<int>(token.token_id));
      if (isFinal) delta += decoder->flush();

      // Record token-level metrics before reasoning filtering so all
      // generated tokens (including hidden reasoning tokens) are counted.
      tt::metrics::ServerMetrics::instance().onToken(taskId);

      TokenParseResult parseResult{ContentType::ANSWER, delta, true};
      if (reasoning_parser_) {
        parseResult = reasoning_parser_->processToken(
            taskId, static_cast<int64_t>(token.token_id), delta);
      }

      if ((!parseResult.should_emit || parseResult.text.empty()) && !isFinal) {
        continue;
      }

      domain::StreamingChunkResponse response{domain::TaskID(taskId)};
      response.id = taskId;
      response.created =
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();

      domain::CompletionChoice choice;
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
              STOP_TOKEN_SET.count(static_cast<int64_t>(token.token_id)) > 0;
          choice.finish_reason = isStop ? "stop" : "length";
        }
      }
      response.choices.push_back(std::move(choice));

      callback(response, isFinal);

      if (isFinal) {
        streamDecoders.erase(taskId);
        std::string finish_reason = "unknown";
        if (!response.choices.empty() &&
            response.choices[0].finish_reason.has_value()) {
          finish_reason = response.choices[0].finish_reason.value();
        }
        tt::metrics::ServerMetrics::instance().onRequestCompleted(
            taskId, finish_reason);
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

domain::CompletionResponse LLMService::processRequest(
    domain::CompletionRequest request) {
  ZoneScopedN("LLMService::processRequest");

  std::mutex mtx;
  std::condition_variable cv;
  bool done = false;

  std::string accumulatedAnswer;
  std::string accumulatedReasoning;
  int completionTokens = 0;
  std::string finishReason = "stop";

  const int PROMPT_TOKENS =
      std::holds_alternative<std::vector<int>>(request.prompt)
          ? static_cast<int>(std::get<std::vector<int>>(request.prompt).size())
          : 0;
  const std::string TASK_ID = request.task_id.id;
  const std::string MODEL = request.model.value_or("default");

  processStreamingRequest(
      std::move(request),
      [&](domain::StreamingChunkResponse& chunk, bool isFinal) {
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

  domain::CompletionResponse response{domain::TaskID(TASK_ID)};
  response.id = TASK_ID;
  response.model = MODEL;
  response.created = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();

  domain::CompletionChoice choice;
  choice.text = std::move(accumulatedAnswer);
  choice.reasoning =
      accumulatedReasoning.empty()
          ? std::nullopt
          : std::optional<std::string>(std::move(accumulatedReasoning));
  choice.index = 0;
  choice.finish_reason = finishReason;
  response.choices.push_back(std::move(choice));

  response.usage = {PROMPT_TOKENS, completionTokens,
                    PROMPT_TOKENS + completionTokens, std::nullopt,
                    std::nullopt};

  return response;
}

void LLMService::processStreamingRequest(
    domain::CompletionRequest request,
    std::function<void(domain::StreamingChunkResponse&, bool isFinal)>
        callback) {
  assert(callback != nullptr);

  ZoneScopedN("LLMService::processStreamingRequest");
  if (request.task_id.id.empty()) {
    throw std::runtime_error("task_id must be set before submitting request");
  }
  std::string taskId = request.task_id.id;

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
      llm_engine::TaskID(taskId),
      tt::config::llmEngineConfig().kvcache_block_size, std::move(tokenIds));
  sequence->numPromptTokens = prompt.size();
  sequence->samplingParams = std::make_unique<llm_engine::SamplingParams>(
      tt::utils::mapper::mapSamplingParams(request));
  queue_manager_->task_queue->push(*std::move(sequence));
}

void LLMService::postProcess(domain::CompletionResponse& response) const {
  // Parse and strip reasoning blocks from all choices
  if (reasoning_parser_) {
    for (auto& choice : response.choices) {
      auto result = reasoning_parser_->parseComplete(choice.text);

      // Replace text with answer only (reasoning stripped)
      choice.text = std::move(result.answer);
    }
  }
}
}  // namespace tt::services
