// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "services/embedding_service.hpp"

#include <poll.h>
#include <signal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "config/defaults.hpp"
#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "services/embedding_codec.hpp"
#include "services/embedding_pipe.hpp"
#include "services/embedding_worker_main.hpp"
#include "services/embedding_worker_process.hpp"
#include "utils/logger.hpp"

namespace tt::services {

using embedding_detail::pipeReadBinary;
using embedding_detail::WorkerProcess;

struct EmbeddingService::Impl {
  struct PendingRequest {
    domain::EmbeddingRequest request;

    std::function<void(domain::EmbeddingResponse&&)> onComplete;

    // Anchors the batch-fill deadline in collectBatch.
    std::chrono::steady_clock::time_point enqueueTime;

    PendingRequest(domain::EmbeddingRequest req,
                   std::function<void(domain::EmbeddingResponse&&)> complete)
        : request(std::move(req)),
          onComplete(std::move(complete)),
          enqueueTime(std::chrono::steady_clock::now()) {}
  };

  static void completeWithError(PendingRequest& p, const std::string& error) {
    domain::EmbeddingResponse err(p.request.task_id);
    err.error = error;
    p.onComplete(std::move(err));
  }

  std::vector<std::unique_ptr<WorkerProcess>> workers;

  mutable std::mutex workersMutex;
  size_t numWorkers = 3;

  TRACY_LOCKABLE(std::mutex, queueMutex);
  std::queue<std::shared_ptr<PendingRequest>> requestQueue;
  std::condition_variable_any queueCv;

  std::atomic<bool> running{false};
  std::atomic<bool> isReady{false};

  // Spawning/warmup run here so start() returns immediately.
  std::unique_ptr<std::thread> startupThread;

  size_t maxBatchSize = 1;
  std::chrono::milliseconds batchTimeout{5};
  size_t maxQueueSize = tt::config::defaults::MAX_QUEUE_SIZE;

  Impl() {
    numWorkers = tt::config::numWorkers();
    maxBatchSize = tt::config::embeddingEngineConfig().max_batch_size;
    batchTimeout = std::chrono::milliseconds(tt::config::batchTimeoutMs());
    maxQueueSize = tt::config::maxQueueSize();
    TT_LOG_INFO(
        "[EmbeddingService] Initialized with {} workers, batch_size={}, "
        "batch_timeout={}ms",
        numWorkers, maxBatchSize, batchTimeout.count());
  }

  ~Impl() { stop(); }

  void start() {
    if (running.exchange(true)) return;

    TT_LOG_INFO("[EmbeddingService] Starting with {} worker processes",
                numWorkers);
    {
      std::lock_guard lock(workersMutex);
      workers.reserve(numWorkers);
      for (size_t i = 0; i < numWorkers; ++i) {
        auto w = std::make_unique<WorkerProcess>();
        w->workerId = static_cast<int>(i);
        workers.push_back(std::move(w));
      }
    }

    startupThread = std::make_unique<std::thread>(&Impl::runStartup, this);
  }

  // The first worker warms up alone so it can populate the shared tensor
  // cache without racing; the rest then warm up in parallel.
  void runStartup() {
    const unsigned warmupTimeoutMs = tt::config::embeddingWarmupTimeoutMs();
    const size_t next = warmupCacheLeader(warmupTimeoutMs);
    spawnAndAwaitRemaining(next, warmupTimeoutMs);
    logStartupSummary();
  }

  // Returns the index of the first worker not yet spawned.
  size_t warmupCacheLeader(unsigned timeoutMs) {
    size_t next = 0;
    bool haveReadyWorker = false;
    while (!haveReadyWorker && next < numWorkers && running.load()) {
      const size_t idx = next++;
      if (!spawnWorkerAt(idx)) continue;
      awaitWorkersReady({idx}, timeoutMs);
      if (workers[idx]->isReady.load()) {
        isReady = true;
        haveReadyWorker = true;
      } else {
        TT_LOG_ERROR(
            "[EmbeddingService] Worker {} failed warmup; trying next worker "
            "alone",
            idx);
      }
    }
    return next;
  }

  void spawnAndAwaitRemaining(size_t firstIdx, unsigned timeoutMs) {
    for (size_t i = firstIdx; i < numWorkers && running.load(); ++i) {
      spawnWorkerAt(i);
    }
    std::vector<size_t> spawned;
    for (size_t i = firstIdx; i < numWorkers; ++i) {
      if (workers[i]->pid.load() > 0) spawned.push_back(i);
    }
    awaitWorkersReady(std::move(spawned), timeoutMs);
  }

  void logStartupSummary() const {
    size_t readyCount = 0;
    for (const auto& w : workers) {
      if (w->isReady.load()) ++readyCount;
    }
    TT_LOG_INFO("[EmbeddingService] Startup finished: {}/{} workers ready",
                readyCount, numWorkers);
  }

  // poll()s all response pipes until every listed worker resolves or the
  // timeout expires.
  void awaitWorkersReady(std::vector<size_t> pending, unsigned timeoutMs) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);

    while (!pending.empty() && running.load()) {
      const auto remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              deadline - std::chrono::steady_clock::now())
              .count();
      if (remaining <= 0) break;

      std::vector<struct pollfd> pfds;
      pfds.reserve(pending.size());
      for (size_t i : pending) {
        pfds.push_back({workers[i]->readFd.get(), POLLIN, 0});
      }

      // 100ms slices keep the shutdown check responsive.
      const int rc = poll(pfds.data(), pfds.size(),
                          static_cast<int>(std::min<int64_t>(remaining, 100)));
      if (rc < 0) {
        if (errno == EINTR) continue;
        TT_LOG_ERROR("[EmbeddingService] Warmup poll failed: {}",
                     strerror(errno));
        break;
      }
      if (rc == 0) continue;

      std::vector<size_t> stillPending;
      for (size_t k = 0; k < pfds.size(); ++k) {
        const size_t i = pending[k];
        if (!resolveWarmupEvent(i, pfds[k].revents)) stillPending.push_back(i);
      }
      pending = std::move(stillPending);
    }

    for (size_t i : pending) {
      TT_LOG_ERROR(
          "[EmbeddingService] Worker {} stuck in warmup after {}ms; "
          "terminating it",
          i, timeoutMs);
      workers[i]->terminate();
    }
  }

  // Returns true once the worker is resolved: ready, or dead and terminated.
  bool resolveWarmupEvent(size_t workerIdx, short revents) {
    if (!(revents & (POLLIN | POLLHUP | POLLERR))) return false;

    if (revents & POLLIN) {
      const auto msg = pipeReadBinary(workers[workerIdx]->readFd.get());
      if (embedding_detail::isReadySentinel(msg)) {
        workers[workerIdx]->isReady.store(true);
        TT_LOG_INFO("[EmbeddingService] Worker {} reported ready", workerIdx);
        launchDispatchThread(workerIdx);
        return true;
      }
      TT_LOG_ERROR(
          "[EmbeddingService] Worker {} exited or sent unexpected data "
          "during warmup",
          workerIdx);
    } else {
      TT_LOG_ERROR(
          "[EmbeddingService] Worker {} closed its pipe during warmup "
          "(process exited)",
          workerIdx);
    }
    workers[workerIdx]->terminate();
    return true;
  }

  bool spawnWorkerAt(size_t idx) {
    const int wid = static_cast<int>(idx);
    return workers[idx]->spawn(wid, [wid](int rd, int wr) {
      embedding_detail::workerProcessMain(wid, rd, wr);
    });
  }

  void launchDispatchThread(size_t idx) {
    workers[idx]->dispatchThread =
        std::make_unique<std::thread>(&Impl::workerDispatchLoop, this, idx);
  }

  std::vector<tt::worker::WorkerInfo> workerInfoSnapshot() const {
    std::lock_guard lock(workersMutex);
    std::vector<tt::worker::WorkerInfo> out;
    out.reserve(workers.size());
    for (const auto& w : workers) {
      if (!w) continue;
      tt::worker::WorkerInfo info;
      info.worker_id = std::to_string(w->workerId);
      info.pid = w->pid.load();
      // kill(pid, 0) probes without reaping (checkAlive/terminate own waitpid)
      info.is_alive = info.pid > 0 && kill(info.pid, 0) == 0;
      info.is_ready = w->isReady.load();
      out.push_back(std::move(info));
    }
    return out;
  }

  void stop() {
    if (!running.exchange(false)) return;

    TT_LOG_INFO("[EmbeddingService] Stopping...");
    if (startupThread && startupThread->joinable()) startupThread->join();
    startupThread.reset();
    queueCv.notify_all();

    for (auto& w : workers) w->running = false;
    queueCv.notify_all();

    for (auto& w : workers) {
      if (w->dispatchThread && w->dispatchThread->joinable())
        w->dispatchThread->join();
      w->terminate();
    }
    // No consumers remain; fail anything still queued.
    drainQueue("Server shutting down");
    {
      std::lock_guard lock(workersMutex);
      workers.clear();
    }
    isReady = false;
    TT_LOG_INFO("[EmbeddingService] Stopped");
  }

  // Fails every queued request; callbacks run outside the queue lock.
  void drainQueue(const std::string& error) {
    std::queue<std::shared_ptr<PendingRequest>> drained;
    {
      std::lock_guard lock(queueMutex);
      std::swap(drained, requestQueue);
    }
    while (!drained.empty()) {
      completeWithError(*drained.front(), error);
      drained.pop();
    }
  }

  // Rolling dispatch-loop statistics; logs a summary every 10 batches.
  struct DispatchStats {
    uint64_t batches = 0;
    uint64_t requests = 0;
    double queueWaitMs = 0;
    double dispatchMs = 0;

    void record(size_t workerIdx, size_t batchSize, double qMs, double dMs) {
      queueWaitMs += qMs;
      batches++;
      requests += batchSize;
      dispatchMs += dMs;

      if (batches % 10 == 0) {
        double avgQueue = queueWaitMs / batches;
        double avgDispatch = dispatchMs / batches;
        double throughput = (requests * 1000.0) / (queueWaitMs + dispatchMs);
        TT_LOG_DEBUG(
            "[EmbeddingService] Worker {} batches={} requests={} "
            "avg_queue_wait={}ms avg_dispatch={}ms throughput={} req/s",
            workerIdx, batches, requests, avgQueue, avgDispatch, throughput);
      }
    }
  };

  // Takes up to maxBatchSize requests; an incomplete batch lingers until
  // batchTimeout past the oldest request's arrival.
  std::vector<std::shared_ptr<PendingRequest>> collectBatch(
      WorkerProcess& worker) {
    std::vector<std::shared_ptr<PendingRequest>> batch;
    std::unique_lock lock(queueMutex);
    queueCv.wait_for(lock, std::chrono::milliseconds(100), [this, &worker] {
      return !requestQueue.empty() || !worker.running.load() || !worker.isReady;
    });

    if (!worker.running.load() || !worker.isReady) return batch;
    if (requestQueue.empty()) return batch;

    if (maxBatchSize > 1 && batchTimeout.count() > 0) {
      while (requestQueue.size() < maxBatchSize) {
        const auto deadline = requestQueue.front()->enqueueTime + batchTimeout;
        if (std::chrono::steady_clock::now() >= deadline) break;
        queueCv.wait_until(lock, deadline, [this, &worker] {
          return requestQueue.size() >= maxBatchSize || requestQueue.empty() ||
                 !worker.running.load() || !worker.isReady;
        });
        if (!worker.running.load() || !worker.isReady) break;
        if (requestQueue.empty()) break;
      }
      if (!worker.running.load() || !worker.isReady) return batch;
      if (requestQueue.empty()) return batch;
    }

    while (batch.size() < maxBatchSize && !requestQueue.empty()) {
      batch.push_back(requestQueue.front());
      requestQueue.pop();
    }
    return batch;
  }

  // If no ready worker remains, queued callbacks would never fire — drain.
  // During stop() workers stay marked ready, so stop()'s own drain handles it.
  void drainIfLastWorker() {
    bool anyReady = false;
    {
      std::lock_guard lock(workersMutex);
      for (const auto& w : workers) {
        if (w && w->isReady.load()) {
          anyReady = true;
          break;
        }
      }
    }
    if (!anyReady) drainQueue("No workers available");
  }

  void workerDispatchLoop(size_t workerIdx) {
    auto& worker = workers[workerIdx];
    TT_LOG_INFO("[EmbeddingService] Worker {} dispatch thread started",
                workerIdx);

    DispatchStats stats;

    while (worker->running.load() && worker->isReady) {
      const auto queueStart = std::chrono::steady_clock::now();
      auto batch = collectBatch(*worker);
      const auto queueEnd = std::chrono::steady_clock::now();

      if (batch.empty()) continue;

      if (!worker->isReady) {
        failBatch(batch, "Worker died");
        continue;
      }

      const auto dispatchStart = std::chrono::steady_clock::now();
      dispatchBatchToWorker(*worker, batch);
      const auto dispatchEnd = std::chrono::steady_clock::now();

      stats.record(
          workerIdx, batch.size(),
          std::chrono::duration<double, std::milli>(queueEnd - queueStart)
              .count(),
          std::chrono::duration<double, std::milli>(dispatchEnd - dispatchStart)
              .count());
    }

    TT_LOG_INFO(
        "[EmbeddingService] Worker {} dispatch thread exiting (isReady={})",
        workerIdx, worker->isReady.load());

    drainIfLastWorker();
  }

  static std::string encodeBatchJson(
      const std::vector<std::shared_ptr<PendingRequest>>& batch) {
    Json::Value batchJson(Json::arrayValue);
    for (const auto& p : batch) batchJson.append(p->request.toJson());
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, batchJson);
  }

  void dispatchBatchToWorker(
      WorkerProcess& worker,
      std::vector<std::shared_ptr<PendingRequest>>& batch) {
    if (!worker.isReady.load() || !worker.checkAlive()) {
      failBatch(batch, "Worker not available");
      return;
    }

    if (!worker.sendRequest(encodeBatchJson(batch))) {
      failBatch(batch, "Worker pipe broken");
      return;
    }

    auto responseBuf = worker.receiveResponse();
    if (responseBuf.empty()) {
      failBatch(batch, "Failed to read response from worker");
      return;
    }

    auto responseMap = embedding_codec::decodeResponses(responseBuf);

    for (auto& pending : batch) {
      auto it = responseMap.find(pending->request.task_id);
      if (it != responseMap.end()) {
        pending->onComplete(std::move(it->second));
      } else {
        completeWithError(*pending, "Response not found for task_id");
      }
    }
  }

  static void failBatch(std::vector<std::shared_ptr<PendingRequest>>& batch,
                        const std::string& error) {
    for (auto& p : batch) completeWithError(*p, error);
  }

  void submitRequestAsync(
      domain::EmbeddingRequest request,
      std::function<void(domain::EmbeddingResponse&&)> onComplete) {
    auto pending = std::make_shared<PendingRequest>(std::move(request),
                                                    std::move(onComplete));
    {
      std::lock_guard lock(queueMutex);
      requestQueue.push(pending);
    }
    queueCv.notify_all();
  }
};

EmbeddingService::EmbeddingService() : impl_(std::make_unique<Impl>()) {
  maxQueueSize = impl_->maxQueueSize;
}

EmbeddingService::~EmbeddingService() = default;

void EmbeddingService::start() { impl_->start(); }

void EmbeddingService::stop() { impl_->stop(); }

bool EmbeddingService::isModelReady() const { return impl_->isReady.load(); }

size_t EmbeddingService::currentQueueSize() const {
  std::lock_guard lock(impl_->queueMutex);
  return impl_->requestQueue.size();
}

std::vector<tt::worker::WorkerInfo> EmbeddingService::getWorkerInfo() const {
  return impl_->workerInfoSnapshot();
}

void EmbeddingService::submitRequestAsync(
    domain::EmbeddingRequest request,
    std::function<void(domain::EmbeddingResponse&&)> onComplete) {
  preProcess(request);
  impl_->submitRequestAsync(std::move(request), std::move(onComplete));
}

}  // namespace tt::services
