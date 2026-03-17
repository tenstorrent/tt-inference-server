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
#include "profiling/tracy.hpp"
#include "utils/logger.hpp"
#include "utils/mapper.hpp"
#include "utils/tokenizer.hpp"
#include "worker/single_process_worker.hpp"

namespace tt::services {

namespace {

[[noreturn]] void execWorkerProcess(
    size_t workerId,
    const std::unordered_map<std::string, std::string>& envVars) {
  for (const auto& [key, value] : envVars) {
    setenv(key.c_str(), value.c_str(), 1);
  }
  char exePath[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
  if (n <= 0) {
    perror("readlink /proc/self/exe");
    _exit(1);
  }
  exePath[n] = '\0';
  char idBuf[16];
  std::snprintf(idBuf, sizeof(idBuf), "%zu", workerId);
  char* execArgv[] = {exePath, const_cast<char*>("--worker"), idBuf, nullptr};
  execv(exePath, execArgv);
  perror("execv");
  _exit(1);
}

}  // namespace

worker::WorkerConfig makeWorkerConfigForProcess(int workerId) {
  worker::WorkerConfig cfg;
  cfg.env_vars["TT_VISIBLE_DEVICES"] =
      tt::config::visibleDevicesForWorker(workerId);
  cfg.task_queue =
      std::make_shared<tt::ipc::BoostIpcTaskQueue>(tt::ipc::TASK_QUEUE_NAME);
  cfg.result_queue =
      std::make_shared<tt::ipc::TokenRingBuffer<tt::ipc::RING_BUFFER_CAPACITY>>(
          "/tt_tokens_" + std::to_string(workerId), false);
  cfg.worker_id = workerId;
  cfg.runner_config = tt::config::llmEngineConfig();
  return cfg;
}

LLMService::LLMService()
    : mode(tt::config::llmMode()),
      numWorkers(tt::config::numWorkers()),
      tokenizer(&tt::utils::activeTokenizer()) {
  max_queue_size_ = tt::config::maxQueueSize();
  TT_LOG_INFO("[LLMService] Initialized (mode={}, workers={})",
              tt::config::to_string(mode), numWorkers);
  queueManager = std::make_unique<tt::ipc::QueueManager>(numWorkers);
  socketService = std::make_shared<tt::sockets::InterServerService>();
  socketService->initializeFromConfig();
}

LLMService::~LLMService() { stop(); }

void LLMService::start() {
  ZoneScopedN("LLMService::start");
  if (running.exchange(true)) {
    return;  // Already running
  }

  TT_LOG_INFO("[LLMService] Starting (mode={}, workers={})",
              tt::config::to_string(mode), numWorkers);

  startWorkers();
  tracy_config::TracyStartupSchedulerParent();
  startConsumers();

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  if (socketService && socketService->isEnabled()) {
    socketService->start();
  }

  isReady = true;
  TracyPlot("pending_tasks", static_cast<double>(pendingTasks.load()));
  TT_LOG_INFO("[LLMService] Service started");
}

bool LLMService::is_model_ready() const { return isReady.load(); }

size_t LLMService::current_queue_size() const { return pendingTasks.load(); }

void LLMService::pre_process(domain::CompletionRequest& request) const {
  BaseService::pre_process(request);
  if (std::holds_alternative<std::string>(request.prompt)) {
    auto text = std::get<std::string>(request.prompt);
    static auto cfg = tt::utils::getTokenizerConfig();
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
  // Set prompt token count after tokenization
  request.prompt_tokens_count = static_cast<int>(tokens.size());
}

void LLMService::startWorkers() {
  for (size_t i = 0; i < numWorkers; i++) {
    tt::worker::WorkerConfig cfg =
        makeWorkerConfigForProcess(static_cast<int>(i));
    workers.push_back(std::make_unique<tt::worker::SingleProcessWorker>(cfg));
    auto& worker = workers[i];

    pid_t pid = fork();

    if (pid < 0) {
      throw std::runtime_error("Failed to fork worker process");
    }
    if (pid == 0) {
      setpgid(0, 0);
      try {
        execWorkerProcess(i, cfg.env_vars);
      } catch (const std::exception& e) {
        TT_LOG_ERROR("[LLMService] Worker {} failed: {}", i, e.what());
        _exit(1);
      }
    }
    setpgid(pid, pid);
    worker->pid = pid;
    TT_LOG_INFO("[LLMService] Spawned worker {} with PID {}", i, pid);
  }
}

void LLMService::startConsumers() {
  consumerThreads.reserve(numWorkers);
  for (size_t i = 0; i < numWorkers; i++) {
    consumerThreads.emplace_back(&LLMService::consumerLoopForWorker, this, i);
  }
  TT_LOG_INFO("[LLMService] Started {} consumer threads", numWorkers);
}

void LLMService::stop() {
  ZoneScopedN("LLMService::stop");
  if (!running.exchange(false)) {
    return;
  }

  TT_LOG_INFO("[LLMService] Stopping...");

  // Signal shutdown on all ring buffers so blocking_pop wakes up
  for (auto& q : queueManager->result_queues) {
    q->shutdown();
  }

  for (auto& thread : consumerThreads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  consumerThreads.clear();

  // Signal shutdown to all workers
  for (auto& w : workers) {
    w->stop();
  }

  workers.clear();

  // Stop socket service
  if (socketService) {
    socketService->stop();
  }

  isReady = false;
  TT_LOG_INFO("[LLMService] Stopped");
  queueManager->clear();
}

bool LLMService::checkWorkerAlive(size_t workerIdx) {
  auto* worker = workers[workerIdx].get();
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
  tracy_config::TracySetThreadName(
      ("Consumer-" + std::to_string(workerIdx)).c_str());

  TT_LOG_INFO("[Consumer-{}] Started", workerIdx);

  auto* worker = workers[workerIdx].get();
  if (!worker->cfg.result_queue) {
    TT_LOG_WARN("[Consumer-{}] No token buffer, exiting", workerIdx);
    return;
  }

  const auto STOP_IDS = tokenizer->stopTokenIds();
  const std::unordered_set<int64_t> STOP_TOKEN_SET(STOP_IDS.begin(),
                                                   STOP_IDS.end());

  while (running) {
    if (!checkWorkerAlive(workerIdx)) {
      TT_LOG_ERROR("[Consumer-{}] Worker process died, exiting consumer",
                   workerIdx);
      break;
    }

    bool anyActivity = false;

    ipc::SharedToken token;
    while (worker->cfg.result_queue->blocking_pop(token)) {
      anyActivity = true;

      auto val = streamCallbacks.get(token.task_id);
      if (!val.has_value()) {
        throw std::runtime_error("callback not found for task_id: " +
                                 std::string(token.task_id));
      }
      auto callback = val.value();
      if (token.is_final()) {
        streamCallbacks.erase(token.task_id);
        pendingTasks.fetch_sub(1);
      }

      domain::StreamingChunkResponse response(
          domain::TaskID(std::string(token.task_id)));
      response.id = std::string(token.task_id);
      response.created =
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::system_clock::now().time_since_epoch())
              .count();

      domain::CompletionChoice choice;
      choice.text = tokenizer->decode({static_cast<int>(token.token_id)});
      choice.index = token.token_index;
      if (token.is_error()) {
        choice.finish_reason = "error";
      } else {
        choice.token_id = static_cast<int64_t>(token.token_id);
        if (token.is_final()) {
          bool isStop =
              STOP_TOKEN_SET.count(static_cast<int64_t>(token.token_id)) > 0;
          choice.finish_reason = isStop ? "stop" : "length";
        }
      }
      response.choices.push_back(std::move(choice));

      callback(response, token.is_final());
      if (token.is_final()) {
        TracyPlot("pending_tasks", static_cast<double>(pendingTasks.load()));
      }
    }

    if (!anyActivity) {
      std::this_thread::yield();
    }
  }

  TT_LOG_INFO("[Consumer-{}] Stopped", workerIdx);
}

domain::CompletionResponse LLMService::process_request(
    domain::CompletionRequest request) {
  ZoneScopedN("LLMService::process_request");

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

  process_streaming_request(
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

void LLMService::process_streaming_request(
    domain::CompletionRequest request,
    std::function<void(domain::StreamingChunkResponse&, bool isFinal)>
        callback) {
  assert(callback != nullptr);

  ZoneScopedN("LLMService::process_streaming_request");
  if (request.task_id.id.empty()) {
    throw std::runtime_error("task_id must be set before submitting request");
  }
  std::string taskId = request.task_id.id;

  pendingTasks.fetch_add(1);
  TracyPlot("pending_tasks", static_cast<double>(pendingTasks.load()));

  streamCallbacks.insert(taskId, std::move(callback));

  auto prompt = std::get<std::vector<int>>(request.prompt);
  std::vector<int64_t> tokenIds(prompt.begin(), prompt.end());

  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    if (!prefillRequestCallback) {
      streamCallbacks.erase(taskId);
      pendingTasks.fetch_sub(1);
      throw std::runtime_error("No prefill request callback configured");
    }

    domain::PrefillRequest prefillReq{domain::TaskID(taskId)};
    prefillReq.token_ids = tokenIds;
    prefillReq.max_tokens = request.max_tokens;

    bool sent = prefillRequestCallback(prefillReq);

    if (!sent) {
      streamCallbacks.erase(taskId);
      pendingTasks.fetch_sub(1);
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
  sequence->numPromptTokens_ = prompt.size();
  sequence->sampling_params = std::make_unique<llm_engine::SamplingParams>(
      tt::utils::mapper::map_sampling_params(request));
  queueManager->task_queue->push(*std::move(sequence));
}

void LLMService::post_process(domain::CompletionResponse&) const {
  // no-op
}

std::shared_ptr<tt::sockets::InterServerService> LLMService::getSocketService()
    const {
  return socketService;
}

void LLMService::setPrefillRequestCallback(PrefillRequestCallback callback) {
  prefillRequestCallback = std::move(callback);
}

std::optional<LLMService::StreamCallback> LLMService::detachStreamCallback(
    const std::string& taskId) {
  auto val = streamCallbacks.take(taskId);
  if (val.has_value()) {
    pendingTasks.fetch_sub(1);
  }
  return val;
}

void LLMService::submitDecodeContinuation(domain::CompletionRequest request,
                                          StreamCallback callback) {
  std::string taskId = request.task_id.id;

  pendingTasks.fetch_add(1);
  TracyPlot("pending_tasks", static_cast<double>(pendingTasks.load()));
  streamCallbacks.insert(taskId, std::move(callback));

  auto prompt = std::get<std::vector<int>>(request.prompt);
  std::vector<int64_t> tokenIds(prompt.begin(), prompt.end());

  auto sequence = std::make_unique<llm_engine::Sequence>(
      llm_engine::TaskID(taskId),
      tt::config::llmEngineConfig().kvcache_block_size, std::move(tokenIds));
  sequence->numPromptTokens_ = prompt.size();
  sequence->sampling_params = std::make_unique<llm_engine::SamplingParams>(
      tt::utils::mapper::map_sampling_params(request));
  queueManager->task_queue->push(*std::move(sequence));

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

  streamCallbacks.for_each(
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

  streamCallbacks.clear();
  pendingTasks.store(0);
}

}  // namespace tt::services
