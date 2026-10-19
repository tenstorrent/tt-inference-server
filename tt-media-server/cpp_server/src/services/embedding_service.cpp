// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "services/embedding_service.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config/defaults.hpp"
#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "utils/logger.hpp"

namespace tt::services {

struct EmbeddingService::Impl {
  struct PendingEntry {
    Callback onComplete;
    std::chrono::steady_clock::time_point enqueueTime;
    std::chrono::steady_clock::time_point claimTime;
    double shmEnqueueMs = 0.0;
    int claimedByWorker = -1;
  };

  /**
   * Host-path observability: per-phase timings and batch-size counts
   * aggregated across worker consumer threads, reported as ONE summary line at
   * shutdown. The CI bench job parses this line to assert that full batches
   * dominated under load.
   */
  struct DispatchSummary {
    struct Phase {
      const char* name;
      std::vector<double> samplesMs;
    };

    std::mutex mutex;
    Phase phases[5] = {{"queue_wait", {}},
                       {"batch_collect", {}},
                       {"shm_enqueue", {}},
                       {"shm_round_trip", {}},
                       {"completion", {}}};
    std::map<size_t, uint64_t> batchSizeCounts;

    void record(size_t batchSize, double queueWaitMs, double collectMs,
                double enqueueMs, double shmRoundTripMs, double completeMs) {
      std::lock_guard lock(mutex);
      phases[0].samplesMs.push_back(queueWaitMs);
      phases[1].samplesMs.push_back(collectMs);
      phases[2].samplesMs.push_back(enqueueMs);
      phases[3].samplesMs.push_back(shmRoundTripMs);
      phases[4].samplesMs.push_back(completeMs);
      batchSizeCounts[batchSize] += 1;
    }

    static double meanOf(const std::vector<double>& v) {
      if (v.empty()) return 0.0;
      double sum = 0.0;
      for (double x : v) sum += x;
      return sum / static_cast<double>(v.size());
    }

    static double p99Of(std::vector<double> v) {
      if (v.empty()) return 0.0;
      std::sort(v.begin(), v.end());
      const size_t idx =
          (v.size() * 99 + 99) / 100 == 0 ? 0 : (v.size() * 99 + 99) / 100 - 1;
      return v[std::min(idx, v.size() - 1)];
    }

    void logSummary() {
      std::lock_guard lock(mutex);
      if (batchSizeCounts.empty()) return;

      std::ostringstream line;
      uint64_t batches = 0;
      uint64_t requests = 0;
      for (const auto& [size, count] : batchSizeCounts) {
        batches += count;
        requests += count * size;
      }
      line << "[EmbeddingService] Dispatch summary: batches=" << batches
           << " requests=" << requests;
      for (auto& phase : phases) {
        line << " " << phase.name << "_ms=" << meanOf(phase.samplesMs) << "/"
             << p99Of(phase.samplesMs) << "(mean/p99)";
      }
      line << " batches:";
      for (auto it = batchSizeCounts.rbegin(); it != batchSizeCounts.rend();
           ++it) {
        line << " " << it->second << "x" << it->first;
      }
      TT_LOG_INFO("{}", line.str());
    }
  };

  config::EmbeddingConfig config;
  std::unique_ptr<tt::worker::WorkerManager> workerManager;
  std::unique_ptr<tt::ipc::embedding::EmbeddingQueueSet> queueSet;

  TRACY_LOCKABLE(std::mutex, pendingMutex);
  std::unordered_map<uint32_t, PendingEntry> pendingResults;
  std::atomic<size_t> inFlight{0};

  std::vector<std::thread> consumerThreads;
  std::atomic<bool> running{false};
  size_t maxQueueSize = tt::config::defaults::MAX_QUEUE_SIZE;
  size_t maxInputBytes = 0;
  DispatchSummary dispatchSummary;

  Impl(config::EmbeddingConfig cfg,
       std::unique_ptr<tt::worker::WorkerManager> wm,
       std::unique_ptr<tt::ipc::embedding::EmbeddingQueueSet> qs)
      : config(std::move(cfg)),
        workerManager(std::move(wm)),
        queueSet(std::move(qs)),
        maxQueueSize(tt::config::maxQueueSize()),
        maxInputBytes(tt::ipc::embedding::maxInputBytesForTaskMsg(
            queueSet ? queueSet->maxTaskMsgBytes
                     : tt::ipc::embedding::EMBEDDING_MAX_TASK_MSG_BYTES)) {
    TT_LOG_INFO(
        "[EmbeddingService] Initialized with {} workers, max_batch_size={}, "
        "max_queue_size={}, max_input_bytes={}",
        workerManager ? workerManager->numWorkers() : 0, config.max_batch_size,
        maxQueueSize, maxInputBytes);
  }

  ~Impl() { stop(); }

  void start() {
    if (running.exchange(true)) return;
    if (!workerManager || !queueSet) return;

    workerManager->setWorkerDeathCallback(
        [this](size_t workerIdx, pid_t /*pid*/) { onWorkerDied(workerIdx); });

    const size_t n = workerManager->numWorkers();
    consumerThreads.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      consumerThreads.emplace_back([this, i]() { consumeResults(i); });
    }

    workerManager->start();
  }

  void stop() {
    if (!running.exchange(false)) {
      drainPending("Server shutting down");
      if (queueSet) {
        queueSet->clear();
      }
      return;
    }

    TT_LOG_INFO("[EmbeddingService] Stopping...");
    if (workerManager && queueSet && queueSet->taskQueue) {
      for (size_t i = 0; i < workerManager->numWorkers(); ++i) {
        try {
          queueSet->taskQueue->push(
              tt::ipc::embedding::EmbeddingIpcTask::done());
        } catch (const std::exception& e) {
          TT_LOG_WARN("[EmbeddingService] Failed to push done task: {}",
                      e.what());
        }
      }
    }

    if (queueSet) {
      for (auto& rq : queueSet->resultQueues) {
        rq->shutdown();
      }
    }

    for (auto& t : consumerThreads) {
      if (t.joinable()) {
        t.join();
      }
    }
    consumerThreads.clear();

    if (workerManager) {
      workerManager->stop();
    }

    drainPending("Server shutting down");
    dispatchSummary.logSummary();

    if (queueSet) {
      queueSet->clear();
    }
    TT_LOG_INFO("[EmbeddingService] Stopped");
  }

  void submit(domain::EmbeddingRequest request, Callback onComplete) {
    if (request.input.size() > maxInputBytes) {
      domain::EmbeddingResponse err(request.task_id);
      err.error = "Input length (" + std::to_string(request.input.size()) +
                  " bytes) exceeds maximum allowed (" +
                  std::to_string(maxInputBytes) + " bytes)";
      onComplete(std::move(err));
      return;
    }

    const uint32_t taskId = request.task_id;
    const auto enqueueStart = std::chrono::steady_clock::now();
    auto ipcTask = tt::ipc::embedding::EmbeddingIpcTask::fromDomain(request);

    {
      std::lock_guard lock(pendingMutex);
      pendingResults[taskId] =
          PendingEntry{std::move(onComplete), enqueueStart, {}, 0.0, -1};
      inFlight.fetch_add(1, std::memory_order_relaxed);
    }

    try {
      queueSet->taskQueue->push(ipcTask);
      const auto enqueueEnd = std::chrono::steady_clock::now();
      const double enqueueMs =
          std::chrono::duration<double, std::milli>(enqueueEnd - enqueueStart)
              .count();
      std::lock_guard lock(pendingMutex);
      auto it = pendingResults.find(taskId);
      if (it != pendingResults.end()) {
        it->second.shmEnqueueMs = enqueueMs;
      }
    } catch (const std::exception& e) {
      Callback cb;
      {
        std::lock_guard lock(pendingMutex);
        auto it = pendingResults.find(taskId);
        if (it != pendingResults.end()) {
          cb = std::move(it->second.onComplete);
          pendingResults.erase(it);
          inFlight.fetch_sub(1, std::memory_order_relaxed);
        }
      }
      if (cb) {
        domain::EmbeddingResponse err(taskId);
        err.error =
            std::string("Failed to enqueue embedding task: ") + e.what();
        cb(std::move(err));
      }
    }
  }

  void consumeResults(size_t workerIdx) {
    auto& rq = queueSet->resultQueues.at(workerIdx);
    tt::ipc::embedding::EmbeddingIpcResult result;
    size_t currentBatchSize = 0;
    double currentQueueWaitMs = 0.0;
    double currentBatchCollectMs = 0.0;
    double currentBatchEnqueueMs = 0.0;
    double currentBatchCompleteMs = 0.0;
    std::chrono::steady_clock::time_point currentBatchClaimTime{};
    std::chrono::steady_clock::time_point currentBatchFirstResultTime{};

    while (rq->blockingPop(result)) {
      if (result.isClaimed()) {
        const auto now = std::chrono::steady_clock::now();
        double queueWaitMs = 0.0;
        double shmEnqueueMs = 0.0;
        {
          std::lock_guard lock(pendingMutex);
          auto it = pendingResults.find(result.taskId);
          if (it != pendingResults.end()) {
            it->second.claimedByWorker = static_cast<int>(workerIdx);
            it->second.claimTime = now;
            queueWaitMs = std::chrono::duration<double, std::milli>(
                              now - it->second.enqueueTime)
                              .count();
            shmEnqueueMs = it->second.shmEnqueueMs;
          }
        }
        if (result.totalTokens > 0) {
          currentBatchSize = static_cast<size_t>(result.totalTokens);
          currentQueueWaitMs = queueWaitMs;
          currentBatchCollectMs = static_cast<double>(result.batchCollectMs);
          currentBatchEnqueueMs = shmEnqueueMs;
          currentBatchCompleteMs = 0.0;
          currentBatchClaimTime = now;
          currentBatchFirstResultTime = {};
        } else {
          currentBatchEnqueueMs += shmEnqueueMs;
        }
        continue;
      }

      const auto gotResultTime = std::chrono::steady_clock::now();
      if (currentBatchFirstResultTime.time_since_epoch().count() == 0) {
        currentBatchFirstResultTime = gotResultTime;
      }
      Callback cb;
      {
        std::lock_guard lock(pendingMutex);
        auto it = pendingResults.find(result.taskId);
        if (it != pendingResults.end()) {
          cb = std::move(it->second.onComplete);
          pendingResults.erase(it);
          inFlight.fetch_sub(1, std::memory_order_relaxed);
        }
      }
      if (cb) {
        cb(result.toDomain());
        const auto doneTime = std::chrono::steady_clock::now();
        currentBatchCompleteMs +=
            std::chrono::duration<double, std::milli>(doneTime - gotResultTime)
                .count();
        if (result.isBatchEnd() && currentBatchSize > 0) {
          const double shmRoundTripMs =
              currentBatchClaimTime.time_since_epoch().count() > 0
                  ? std::chrono::duration<double, std::milli>(
                        currentBatchFirstResultTime - currentBatchClaimTime)
                        .count()
                  : 0.0;
          dispatchSummary.record(currentBatchSize, currentQueueWaitMs,
                                 currentBatchCollectMs, currentBatchEnqueueMs,
                                 shmRoundTripMs, currentBatchCompleteMs);
          currentBatchSize = 0;
        }
      } else {
        TT_LOG_WARN("[EmbeddingService] Unknown task_id={}", result.taskId);
      }
    }
  }

  void onWorkerDied(size_t workerIdx) {
    if (workerIdx < queueSet->resultQueues.size()) {
      auto& rq = queueSet->resultQueues[workerIdx];
      tt::ipc::embedding::EmbeddingIpcResult result;
      while (rq->tryPop(result)) {
        if (result.isDone()) continue;
        if (result.isClaimed()) {
          std::lock_guard lock(pendingMutex);
          auto it = pendingResults.find(result.taskId);
          if (it != pendingResults.end()) {
            it->second.claimedByWorker = static_cast<int>(workerIdx);
          }
          continue;
        }
        Callback cb;
        {
          std::lock_guard lock(pendingMutex);
          auto it = pendingResults.find(result.taskId);
          if (it != pendingResults.end()) {
            cb = std::move(it->second.onComplete);
            pendingResults.erase(it);
            inFlight.fetch_sub(1, std::memory_order_relaxed);
          }
        }
        if (cb) {
          cb(result.toDomain());
        }
      }
    }

    std::vector<std::pair<uint32_t, Callback>> orphaned;
    {
      std::lock_guard lock(pendingMutex);
      for (auto it = pendingResults.begin(); it != pendingResults.end();) {
        if (it->second.claimedByWorker == static_cast<int>(workerIdx)) {
          orphaned.emplace_back(it->first, std::move(it->second.onComplete));
          it = pendingResults.erase(it);
          inFlight.fetch_sub(1, std::memory_order_relaxed);
        } else {
          ++it;
        }
      }
    }

    for (auto& [taskId, cb] : orphaned) {
      if (cb) {
        domain::EmbeddingResponse err(taskId);
        err.error = "Worker process " + std::to_string(workerIdx) +
                    " exited while processing request";
        cb(std::move(err));
      }
    }

    if (running.load(std::memory_order_acquire) && workerManager &&
        workerManager->isReady()) {
      // At least one live ready worker remains; unclaimed tasks will be picked
      // up by surviving workers.
      return;
    }

    // If startup already finished and the last ready worker just died, fail
    // any remaining unclaimed tasks so callers don't hang forever.
    if (workerManager) {
      bool anyAliveAndReady = false;
      for (const auto& info : workerManager->getWorkerInfo()) {
        if (info.is_alive && info.is_ready) {
          anyAliveAndReady = true;
          break;
        }
      }
      if (!anyAliveAndReady && !orphaned.empty()) {
        drainPending("No workers available");
      }
    }
  }

  void drainPending(const std::string& error) {
    std::vector<std::pair<uint32_t, Callback>> remaining;
    {
      std::lock_guard lock(pendingMutex);
      remaining.reserve(pendingResults.size());
      for (auto& [taskId, entry] : pendingResults) {
        remaining.emplace_back(taskId, std::move(entry.onComplete));
      }
      pendingResults.clear();
      inFlight.store(0, std::memory_order_relaxed);
    }
    for (auto& [taskId, cb] : remaining) {
      if (cb) {
        domain::EmbeddingResponse err(taskId);
        err.error = error;
        cb(std::move(err));
      }
    }
  }
};

namespace {

std::unique_ptr<tt::worker::WorkerManager> makeDefaultEmbeddingWorkerManager() {
  tt::worker::WorkerManagerOptions opts;
  opts.sequentialLeadWorkerWarmup = true;
  opts.warmupTimeoutMs = tt::config::embeddingWarmupTimeoutMs();
  return std::make_unique<tt::worker::WorkerManager>(tt::config::numWorkers(),
                                                     opts);
}

std::unique_ptr<tt::ipc::embedding::EmbeddingQueueSet>
makeDefaultEmbeddingQueueSet(const config::EmbeddingConfig& cfg) {
  return std::make_unique<tt::ipc::embedding::EmbeddingQueueSet>(
      static_cast<int>(tt::config::numWorkers()), cfg.max_seq_len,
      cfg.max_batch_size);
}

}  // namespace

EmbeddingService::EmbeddingService()
    : EmbeddingService(
          config::embeddingEngineConfig(), makeDefaultEmbeddingWorkerManager(),
          makeDefaultEmbeddingQueueSet(config::embeddingEngineConfig())) {}

EmbeddingService::EmbeddingService(
    config::EmbeddingConfig config,
    std::unique_ptr<tt::worker::WorkerManager> workerManager,
    std::unique_ptr<tt::ipc::embedding::EmbeddingQueueSet> queueSet)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(workerManager),
                                   std::move(queueSet))) {
  maxQueueSize = impl_->maxQueueSize;
}

EmbeddingService::~EmbeddingService() = default;

void EmbeddingService::start() { impl_->start(); }

void EmbeddingService::stop() { impl_->stop(); }

bool EmbeddingService::isModelReady() const {
  return impl_->workerManager && impl_->workerManager->isReady();
}

size_t EmbeddingService::currentQueueSize() const {
  return impl_->inFlight.load(std::memory_order_relaxed);
}

std::vector<tt::worker::WorkerInfo> EmbeddingService::getWorkerInfo() const {
  if (!impl_->workerManager) return {};
  return impl_->workerManager->getWorkerInfo();
}

void EmbeddingService::produceResponseAsync(domain::EmbeddingRequest request,
                                            Callback onComplete) {
  impl_->submit(std::move(request), std::move(onComplete));
}

}  // namespace tt::services
