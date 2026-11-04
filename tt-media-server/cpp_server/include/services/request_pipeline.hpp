// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config/settings.hpp"
#include "domain/base_request.hpp"
#include "domain/base_response.hpp"
#include "runtime/worker/worker_info.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "utils/logger.hpp"

namespace tt::services {

class QueueFullException : public std::runtime_error {
 public:
  QueueFullException()
      : std::runtime_error("Request queue is full, please retry later") {}
};

struct SystemStatus {
  bool modelReady;
  size_t queueSize;
  size_t maxQueueSize;
  std::vector<tt::worker::WorkerInfo> workerInfo;
};

class IService {
 public:
  virtual ~IService() = default;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual bool isModelReady() const = 0;
  virtual bool hasFatalError() const { return false; }
  virtual SystemStatus getSystemStatus() const = 0;
  virtual std::string runnerInUse() const { return ""; }
};

/** Shared queue back-pressure and system status for all request pipelines. */
template <std::derived_from<domain::BaseRequest> RequestType>
class RequestPipeline : public IService {
 public:
  virtual ~RequestPipeline() = default;

  SystemStatus getSystemStatus() const override {
    SystemStatus status;
    status.modelReady = isModelReady();
    status.queueSize = currentQueueSize();
    status.maxQueueSize = maxQueueSize;
    status.workerInfo = getWorkerInfo();
    return status;
  }

  bool isModelReady() const override { return false; }

 protected:
  virtual void preProcess(RequestType& /*request*/) const {
    enforceQueueCapacity();
  }

  void enforceQueueCapacity() const {
    if (currentQueueSize() >= maxQueueSize) {
      throw QueueFullException{};
    }
  }

  /** Override when the service has its own queue; default is no back-pressure.
   */
  virtual size_t currentQueueSize() const { return 0; }

  virtual std::vector<tt::worker::WorkerInfo> getWorkerInfo() const {
    return {};
  }

  size_t maxQueueSize = std::numeric_limits<size_t>::max();
};

/**
 * Worker-backed single-response async service base class (Template Method).
 *
 * Owns WorkerManager, shared-memory QueueSet, per-worker result consumer
 * threads, in-flight request tracking, and crash attribution (FLAG_CLAIMED).
 * Subclasses override hooks to translate between domain and IPC types.
 */
template <std::derived_from<domain::BaseRequest> RequestType,
          std::derived_from<domain::BaseResponse> ResponseType,
          typename QueueSetType>
class BaseAsyncService : public RequestPipeline<RequestType> {
 public:
  using Callback = std::function<void(ResponseType&&)>;
  using IpcTask = typename QueueSetType::TaskType;
  using IpcResult = typename QueueSetType::ResultType;

  BaseAsyncService(std::string serviceName,
                   std::unique_ptr<tt::worker::WorkerManager> workerManager,
                   std::unique_ptr<QueueSetType> queueSet)
      : serviceName_(std::move(serviceName)),
        workerManager_(std::move(workerManager)),
        queueSet_(std::move(queueSet)) {
    this->maxQueueSize = tt::config::maxQueueSize();
  }

  ~BaseAsyncService() override = default;

  BaseAsyncService(const BaseAsyncService&) = delete;
  BaseAsyncService& operator=(const BaseAsyncService&) = delete;

  void start() override {
    if (running_.exchange(true, std::memory_order_acq_rel)) return;
    if (!workerManager_ || !queueSet_) return;

    {
      std::lock_guard<std::mutex> lock(failureMutex_);
      failureReason_.clear();
    }
    workerFailed_.store(false, std::memory_order_release);

    workerManager_->setWorkerDeathCallback(
        [this](size_t workerIdx, pid_t /*pid*/) { onWorkerDied(workerIdx); });

    const size_t n = workerManager_->numWorkers();
    onBeforeStartWorkers(n);
    consumerThreads_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      consumerThreads_.emplace_back([this, i]() { consumerLoopForWorker(i); });
    }

    workerManager_->start();
  }

  void stop() override {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
      drainPending("Server shutting down");
      if (queueSet_) {
        queueSet_->clear();
      }
      onStopped();
      return;
    }

    TT_LOG_INFO("[{}] Stopping...", serviceName_);
    // Non-blocking: the task queue may be full (e.g. requests accepted during
    // warmup when no worker ever drained them). Workers that miss the done
    // task still exit via WorkerManager::stop() and the timed task poll.
    if (workerManager_ && queueSet_ && queueSet_->taskQueue) {
      for (size_t i = 0; i < workerManager_->numWorkers(); ++i) {
        try {
          if (!queueSet_->taskQueue->tryPush(IpcTask::done())) {
            TT_LOG_WARN("[{}] Task queue full; skipping done task",
                        serviceName_);
            break;
          }
        } catch (const std::exception& e) {
          TT_LOG_WARN("[{}] Failed to push done task: {}", serviceName_,
                      e.what());
        }
      }
    }

    if (queueSet_) {
      for (auto& rq : queueSet_->resultQueues) {
        rq->shutdown();
      }
    }

    for (auto& t : consumerThreads_) {
      if (t.joinable()) {
        t.join();
      }
    }
    consumerThreads_.clear();

    if (workerManager_) {
      workerManager_->stop();
    }

    drainPending("Server shutting down");
    onStopped();

    if (queueSet_) {
      queueSet_->clear();
    }
    TT_LOG_INFO("[{}] Stopped", serviceName_);
  }

  bool isModelReady() const override {
    return !workerFailed_.load(std::memory_order_acquire) && workerManager_ &&
           workerManager_->isReady();
  }

  bool hasFatalError() const override {
    return workerFailed_.load(std::memory_order_acquire);
  }

  void submitRequestAsync(RequestType request, Callback onComplete) {
    this->preProcess(request);
    produceResponseAsync(std::move(request), std::move(onComplete));
  }

 protected:
  virtual std::string validateRequest(const RequestType& /*request*/) const {
    return {};
  }

  virtual IpcTask buildIpcTask(const RequestType& request) = 0;

  virtual ResponseType buildResponse(uint32_t taskId,
                                     const IpcResult& result) = 0;

  virtual void cleanupTask(uint32_t /*taskId*/) {}

  virtual void onBeforeStartWorkers(size_t /*numWorkers*/) {}

  virtual void onTaskClaimed(
      size_t /*workerIdx*/, const IpcResult& /*result*/,
      std::chrono::steady_clock::time_point /*claimTime*/,
      double /*queueWaitMs*/, double /*shmEnqueueMs*/) {}

  virtual void onTaskCompleted(
      size_t /*workerIdx*/, const IpcResult& /*result*/,
      std::chrono::steady_clock::time_point /*gotResultTime*/,
      std::chrono::steady_clock::time_point /*doneTime*/) {}

  virtual void onStopped() {}

  size_t currentQueueSize() const override {
    return inFlight_.load(std::memory_order_relaxed);
  }

  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override {
    if (!workerManager_) return {};
    return workerManager_->getWorkerInfo();
  }

  size_t numWorkers() const {
    return workerManager_ ? workerManager_->numWorkers() : 0;
  }

  QueueSetType* queueSet() const { return queueSet_.get(); }

  void produceResponseAsync(RequestType request, Callback onComplete) {
    const uint32_t taskId = request.task_id;
    if (workerFailed_.load(std::memory_order_acquire)) {
      onComplete(makeErrorResponse(taskId, failureReason()));
      return;
    }
    if (std::string err = validateRequest(request); !err.empty()) {
      onComplete(makeErrorResponse(taskId, std::move(err)));
      return;
    }

    IpcTask ipcTask;
    try {
      ipcTask = buildIpcTask(request);
    } catch (const std::exception& e) {
      cleanupTask(taskId);
      onComplete(makeErrorResponse(taskId, e.what()));
      return;
    }

    const auto enqueueStart = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(pendingMutex_);
      pendingResults_[taskId] =
          PendingEntry{std::move(onComplete), enqueueStart, 0.0, -1};
      inFlight_.fetch_add(1, std::memory_order_relaxed);
    }

    try {
      queueSet_->taskQueue->push(ipcTask);
      const auto enqueueEnd = std::chrono::steady_clock::now();
      const double enqueueMs =
          std::chrono::duration<double, std::milli>(enqueueEnd - enqueueStart)
              .count();
      std::lock_guard<std::mutex> lock(pendingMutex_);
      auto it = pendingResults_.find(taskId);
      if (it != pendingResults_.end()) {
        it->second.shmEnqueueMs = enqueueMs;
      }
    } catch (const std::exception& e) {
      Callback cb;
      {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        auto it = pendingResults_.find(taskId);
        if (it != pendingResults_.end()) {
          cb = std::move(it->second.onComplete);
          pendingResults_.erase(it);
          inFlight_.fetch_sub(1, std::memory_order_relaxed);
        }
      }
      cleanupTask(taskId);
      if (cb) {
        cb(makeErrorResponse(
            taskId, std::string("Failed to enqueue task: ") + e.what()));
      }
      return;
    }

    if (workerFailed_.load(std::memory_order_acquire)) {
      drainPending(failureReason());
    }
  }

 private:
  struct PendingEntry {
    Callback onComplete;
    std::chrono::steady_clock::time_point enqueueTime;
    double shmEnqueueMs = 0.0;
    int claimedByWorker = -1;
  };

  static ResponseType makeErrorResponse(uint32_t taskId, std::string error) {
    ResponseType resp(taskId);
    resp.error = std::move(error);
    return resp;
  }

  void consumerLoopForWorker(size_t workerIdx) {
    auto& rq = queueSet_->resultQueues.at(workerIdx);
    IpcResult result;
    while (rq->blockingPop(result)) {
      const uint32_t taskId = result.getTaskId();
      if (result.isClaimed()) {
        const auto now = std::chrono::steady_clock::now();
        double queueWaitMs = 0.0;
        double shmEnqueueMs = 0.0;
        {
          std::lock_guard<std::mutex> lock(pendingMutex_);
          auto it = pendingResults_.find(taskId);
          if (it != pendingResults_.end()) {
            it->second.claimedByWorker = static_cast<int>(workerIdx);
            queueWaitMs = std::chrono::duration<double, std::milli>(
                              now - it->second.enqueueTime)
                              .count();
            shmEnqueueMs = it->second.shmEnqueueMs;
          }
        }
        onTaskClaimed(workerIdx, result, now, queueWaitMs, shmEnqueueMs);
        continue;
      }

      const auto gotResultTime = std::chrono::steady_clock::now();
      Callback cb;
      {
        std::lock_guard<std::mutex> lock(pendingMutex_);
        auto it = pendingResults_.find(taskId);
        if (it != pendingResults_.end()) {
          cb = std::move(it->second.onComplete);
          pendingResults_.erase(it);
          inFlight_.fetch_sub(1, std::memory_order_relaxed);
        }
      }
      if (!cb) {
        cleanupTask(taskId);
        TT_LOG_WARN("[{}] Unknown task_id={}", serviceName_, taskId);
        continue;
      }

      ResponseType response = buildResponse(taskId, result);
      cleanupTask(taskId);
      cb(std::move(response));
      const auto doneTime = std::chrono::steady_clock::now();
      onTaskCompleted(workerIdx, result, gotResultTime, doneTime);
    }
  }

  void onWorkerDied(size_t workerIdx) {
    if (workerIdx < queueSet_->resultQueues.size()) {
      auto& rq = queueSet_->resultQueues[workerIdx];
      IpcResult result;
      while (rq->tryPop(result)) {
        if (result.isDone()) continue;
        const uint32_t taskId = result.getTaskId();
        if (result.isClaimed()) {
          std::lock_guard<std::mutex> lock(pendingMutex_);
          auto it = pendingResults_.find(taskId);
          if (it != pendingResults_.end()) {
            it->second.claimedByWorker = static_cast<int>(workerIdx);
          }
          continue;
        }
        Callback cb;
        {
          std::lock_guard<std::mutex> lock(pendingMutex_);
          auto it = pendingResults_.find(taskId);
          if (it != pendingResults_.end()) {
            cb = std::move(it->second.onComplete);
            pendingResults_.erase(it);
            inFlight_.fetch_sub(1, std::memory_order_relaxed);
          }
        }
        if (cb) {
          ResponseType response = buildResponse(taskId, result);
          cleanupTask(taskId);
          cb(std::move(response));
        }
      }
    }

    const std::string reason = "Worker process " + std::to_string(workerIdx) +
                               " exited while processing request";
    {
      std::lock_guard<std::mutex> lock(failureMutex_);
      if (failureReason_.empty()) {
        failureReason_ = reason;
      }
    }
    workerFailed_.store(true, std::memory_order_release);
    drainPending(reason);
  }

  std::string failureReason() const {
    std::lock_guard<std::mutex> lock(failureMutex_);
    if (!failureReason_.empty()) {
      return failureReason_;
    }
    return "Worker process exited";
  }

  void drainPending(const std::string& error) {
    std::vector<std::pair<uint32_t, Callback>> remaining;
    {
      std::lock_guard<std::mutex> lock(pendingMutex_);
      remaining.reserve(pendingResults_.size());
      for (auto& [taskId, entry] : pendingResults_) {
        remaining.emplace_back(taskId, std::move(entry.onComplete));
      }
      pendingResults_.clear();
      inFlight_.store(0, std::memory_order_relaxed);
    }
    for (auto& [taskId, cb] : remaining) {
      cleanupTask(taskId);
      if (cb) {
        cb(makeErrorResponse(taskId, error));
      }
    }
  }

  std::string serviceName_;
  std::unique_ptr<tt::worker::WorkerManager> workerManager_;
  std::unique_ptr<QueueSetType> queueSet_;
  std::vector<std::thread> consumerThreads_;
  mutable std::mutex pendingMutex_;
  std::unordered_map<uint32_t, PendingEntry> pendingResults_;
  std::atomic<size_t> inFlight_{0};
  std::atomic<bool> running_{false};
  std::atomic<bool> workerFailed_{false};
  mutable std::mutex failureMutex_;
  std::string failureReason_;
};

/** Streaming services: many chunks per request via callback. */
template <std::derived_from<domain::BaseRequest> RequestType,
          std::derived_from<domain::BaseResponse> ChunkType>
class BaseStreamingService : public RequestPipeline<RequestType> {
 public:
  virtual ~BaseStreamingService() = default;

  void submitStreamingRequest(
      RequestType& request,
      std::function<void(const ChunkType&, bool isFinal)> callback,
      bool skipPreProcess = false) {
    if (!skipPreProcess) {
      this->preProcess(request);
    }
    produceStream(std::move(request), [this, cb = std::move(callback)](
                                          ChunkType& chunk, bool isFinal) {
      streamingPostProcess(chunk);
      cb(chunk, isFinal);
    });
  }

  virtual void abortRequest(uint32_t /*taskId*/) {}

 protected:
  virtual void produceStream(
      RequestType request,
      std::function<void(ChunkType&, bool isFinal)> callback) = 0;

  virtual void streamingPostProcess(ChunkType& /*chunk*/) const {}
};

}  // namespace tt::services
