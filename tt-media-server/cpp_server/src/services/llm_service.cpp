// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "services/llm_service.hpp"

#include <sys/wait.h>

#include <cassert>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_set>

#include "config/settings.hpp"
#include "ipc/boost_ipc_warmup_signal_queue.hpp"
#include "profiling/tracy.hpp"
#include "utils/logger.hpp"
#include "utils/mapper.hpp"
#include "utils/tokenizer.hpp"
#include "worker/single_process_worker.hpp"

namespace tt::services {

std::unique_ptr<tt::ipc::IWarmupSignalQueue> LLMService::createWarmupQueue(
    const std::string& name, size_t capacity) {
  tt::ipc::BoostIpcWarmupSignalQueue::remove(name);
  return std::make_unique<tt::ipc::BoostIpcWarmupSignalQueue>(name, capacity);
}

LLMService::LLMService()
    : mode_(tt::config::llmMode()), tokenizer_(&tt::utils::activeTokenizer()) {
  num_workers_ = tt::config::numWorkers();
  max_queue_size_ = tt::config::maxQueueSize();
  TT_LOG_INFO("[LLMService] Initialized (mode={}, workers={})",
              tt::config::toString(mode_), num_workers_);
  queue_manager_ = std::make_unique<tt::ipc::QueueManager>(num_workers_);

  socket_service_ = std::make_shared<tt::sockets::InterServerService>();
  socket_service_->initializeFromConfig();
}

LLMService::~LLMService() { stop(); }

void LLMService::start() {
  ZoneScopedN("LLMService::start");
  if (running_.exchange(true)) {
    return;  // Already running
  }
  if (num_workers_ == 0) {
    running_ = false;
    throw std::invalid_argument(
        "LLMService requires at least one worker (num_workers > 0)");
  }

  TT_LOG_INFO("[LLMService] Starting (mode={}, workers={})",
              tt::config::toString(mode_), num_workers_);

  startWarmupListener(tt::ipc::WARMUP_SIGNALS_QUEUE_NAME, num_workers_);
  startWorkers();
  tracy_config::tracyStartupSchedulerParent();

  waitForFirstWarmup();
  is_ready_ = true;
  startConsumers();

  if (socket_service_ && socket_service_->isEnabled()) {
    socket_service_->start();
  }

  TRACY_PLOT("pending_tasks", static_cast<double>(pending_tasks_.load()));
  TT_LOG_INFO("[LLMService] Service started");
}

bool LLMService::isModelReady() const { return is_ready_.load(); }

size_t LLMService::currentQueueSize() const { return pending_tasks_.load(); }

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
  // Set prompt token count after tokenization
  request.prompt_tokens_count = static_cast<int>(tokens.size());
}

void LLMService::startConsumers() {
  consumer_threads_.reserve(num_workers_);
  for (size_t i = 0; i < num_workers_; i++) {
    consumer_threads_.emplace_back(&LLMService::consumerLoopForWorker, this, i);
  }
  TT_LOG_INFO("[LLMService] Started {} consumer threads", num_workers_);
}

void LLMService::stop() {
  ZoneScopedN("LLMService::stop");
  if (!running_.exchange(false)) {
    return;
  }

  TT_LOG_INFO("[LLMService] Stopping...");

  stopWarmupListener();

  // Signal shutdown on all ring buffers so blockingPop wakes up
  for (auto& q : queue_manager_->result_queues) {
    q->shutdown();
  }

  for (auto& thread : consumer_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  consumer_threads_.clear();

  // Signal shutdown to all workers
  for (auto& w : workers_) {
    w->stop();
  }

  workers_.clear();

  // Stop socket service
  if (socket_service_) {
    socket_service_->stop();
  }

  is_ready_ = false;
  TT_LOG_INFO("[LLMService] Stopped");
  queue_manager_->clear();
}

bool LLMService::checkWorkerAlive(size_t workerIdx) {
  auto* worker = workers_[workerIdx].get();
  if (worker->pid <= 0) {
    return false;
  }

  int status;
  pid_t result = waitpid(worker->pid, &status, WNOHANG);
  if (result == 0) {
    return true;  // Still running
  }
  if (result == worker->pid) {
    worker->is_alive = false;
    return false;
  }
  return true;  // Error in waitpid, assume alive
}

void LLMService::consumerLoopForWorker(size_t workerIdx) {
  ZoneScopedN("LLMService::consumer_loop");
  tracy_config::tracySetThreadName(
      ("Consumer-" + std::to_string(workerIdx)).c_str());

  TT_LOG_INFO("[Consumer-{}] Started", workerIdx);

  auto* worker = workers_[workerIdx].get();
  if (!worker->cfg.result_queue) {
    TT_LOG_WARN("[Consumer-{}] No token buffer, exiting", workerIdx);
    return;
  }

  const auto STOP_IDS = tokenizer_->stopTokenIds();
  const std::unordered_set<int64_t> STOP_TOKEN_SET(STOP_IDS.begin(),
                                                   STOP_IDS.end());

  while (running_) {
    if (!checkWorkerAlive(workerIdx)) {
      TT_LOG_ERROR("[Consumer-{}] Worker process died, exiting consumer",
                   workerIdx);
      break;
    }

    bool anyActivity = false;

    ipc::SharedToken token;
    while (worker->cfg.result_queue->blockingPop(token)) {
      anyActivity = true;

      auto val = stream_callbacks_.get(token.task_id);
      if (!val.has_value()) {
        throw std::runtime_error("callback not found for task_id: " +
                                 std::string(token.task_id));
      }
      auto callback = val.value();
      if (token.isFinal()) {
        stream_callbacks_.erase(token.task_id);
        pending_tasks_.fetch_sub(1);
      }

      domain::StreamingChunkResponse response(
          domain::TaskID(std::string(token.task_id)));
      response.id = std::string(token.task_id);
      response.created =
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();

      domain::CompletionChoice choice;
      choice.text = tokenizer_->decode({static_cast<int>(token.token_id)});
      choice.index = token.token_index;
      if (token.isError()) {
        choice.finish_reason = "error";
      } else {
        choice.token_id = static_cast<int64_t>(token.token_id);
        if (token.isFinal()) {
          bool isStop =
              STOP_TOKEN_SET.count(static_cast<int64_t>(token.token_id)) > 0;
          choice.finish_reason = isStop ? "stop" : "length";
        }
      }
      response.choices.push_back(std::move(choice));

      callback(response, token.isFinal());
      if (token.isFinal()) {
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

  std::string accumulatedText;
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
          accumulatedText.append(chunk.choices[0].text);
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
  choice.text = std::move(accumulatedText);
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

  stream_callbacks_.insert(taskId, std::move(callback));

  auto prompt = std::get<std::vector<int>>(request.prompt);
  std::vector<int64_t> tokenIds(prompt.begin(), prompt.end());

  if (mode_ == tt::config::LLMMode::DECODE_ONLY) {
    if (!prefill_request_callback_) {
      stream_callbacks_.erase(taskId);
      pending_tasks_.fetch_sub(1);
      throw std::runtime_error("No prefill request callback configured");
    }

    domain::PrefillRequest prefillReq{domain::TaskID(taskId)};
    prefillReq.token_ids = tokenIds;
    prefillReq.max_tokens = request.max_tokens;

    bool sent = prefill_request_callback_(prefillReq);

    if (!sent) {
      stream_callbacks_.erase(taskId);
      pending_tasks_.fetch_sub(1);
      throw std::runtime_error(
          "Failed to send prefill request (not connected)");
    }
    TT_LOG_DEBUG("[LLMService:DECODE] Forwarded prefill request {} ({} tokens)",
                 taskId, tokenIds.size());
    return;
  }

  auto sequence = std::make_unique<llm_engine::Sequence>(
      llm_engine::TaskID(taskId),
      tt::config::llmEngineConfig().kvcache_block_size, std::move(tokenIds));
  sequence->num_prompt_tokens_ = prompt.size();
  sequence->sampling_params = std::make_unique<llm_engine::SamplingParams>(
      tt::utils::mapper::mapSamplingParams(request));
  queue_manager_->task_queue->push(*std::move(sequence));
}

void LLMService::postProcess(domain::CompletionResponse&) const {
  // no-op
}

std::shared_ptr<tt::sockets::InterServerService> LLMService::getSocketService()
    const {
  return socket_service_;
}

void LLMService::setPrefillRequestCallback(PrefillRequestCallback callback) {
  prefill_request_callback_ = std::move(callback);
}

std::optional<LLMService::StreamCallback> LLMService::detachStreamCallback(
    const std::string& taskId) {
  auto val = stream_callbacks_.take(taskId);
  if (val.has_value()) {
    pending_tasks_.fetch_sub(1);
  }
  return val;
}

void LLMService::submitDecodeContinuation(domain::CompletionRequest request,
                                          StreamCallback callback) {
  std::string taskId = request.task_id.id;

  pending_tasks_.fetch_add(1);
  TRACY_PLOT("pending_tasks", static_cast<double>(pending_tasks_.load()));
  stream_callbacks_.insert(taskId, std::move(callback));

  auto prompt = std::get<std::vector<int>>(request.prompt);
  std::vector<int64_t> tokenIds(prompt.begin(), prompt.end());

  auto sequence = std::make_unique<llm_engine::Sequence>(
      llm_engine::TaskID(taskId),
      tt::config::llmEngineConfig().kvcache_block_size, std::move(tokenIds));
  sequence->num_prompt_tokens_ = prompt.size();
  sequence->sampling_params = std::make_unique<llm_engine::SamplingParams>(
      tt::utils::mapper::mapSamplingParams(request));
  queue_manager_->task_queue->push(*std::move(sequence));

  TT_LOG_DEBUG(
      "[LLMService:DECODE] Queued decode continuation for task {} "
      "(prompt_tokens={}, max_tokens={})",
      taskId, prompt.size(),
      request.max_tokens.has_value()
          ? std::to_string(request.max_tokens.value())
          : "none");
}

void LLMService::handleConnectionLost() {
  TT_LOG_ERROR("[LLMService] Failing pending tasks due to connection loss");

  stream_callbacks_.forEach(
      [](const std::string& taskId,
         std::function<void(domain::StreamingChunkResponse&, bool)>& callback) {
        domain::StreamingChunkResponse errorResponse{domain::TaskID(taskId)};
        errorResponse.id = taskId;
        errorResponse.created =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();

        domain::CompletionChoice choice;
        choice.text = "";
        choice.index = 0;
        choice.finish_reason = "error";
        errorResponse.choices.push_back(std::move(choice));
        errorResponse.error = "Connection to remote server lost";

        callback(errorResponse, true);
      });

  stream_callbacks_.clear();
  pending_tasks_.store(0);
}

}  // namespace tt::services
