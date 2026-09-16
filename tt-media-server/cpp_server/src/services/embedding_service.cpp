// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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
#include "services/embedding_service.hpp"
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

    std::chrono::steady_clock::time_point enqueueTime;

    PendingRequest(domain::EmbeddingRequest req,
                   std::function<void(domain::EmbeddingResponse&&)> complete)
        : request(std::move(req)),
          onComplete(std::move(complete)),
          enqueueTime(std::chrono::steady_clock::now()) {}
  };

  std::vector<std::unique_ptr<WorkerProcess>> workers;

  mutable std::mutex workersMutex;
  size_t numWorkers = 3;

  TRACY_LOCKABLE(std::mutex, queueMutex);
  std::queue<std::shared_ptr<PendingRequest>> requestQueue;
  std::condition_variable_any queueCv;

  std::atomic<bool> running{false};
  std::atomic<bool> isReady{false};

  // Spawning and warmup run here so start() returns immediately and the HTTP
  // server can answer health probes while the model loads
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

  /**
   * Bring up worker 0 alone and wait for its READY handshake before spawning
   * the rest. The first warmup on a cold volume generates the shared tensor
   * cache. Once one worker has written the cache, the remaining workers warm up
   * in parallel safely.
   */
  void runStartup() {
    const unsigned warmupTimeoutMs = tt::config::embeddingWarmupTimeoutMs();

    // Phase 1: warm up a single worker with exclusive cache access. If it
    // fails, try the next one alone
    size_t next = 0;
    bool haveReadyWorker = false;
    while (!haveReadyWorker && next < numWorkers && running.load()) {
      const size_t idx = next++;
      if (!spawnWorkerAt(idx)) continue;
      awaitWorkersReady({idx}, warmupTimeoutMs);
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

    // Phase 2: the tensor cache is warm; the rest can load concurrently.
    const size_t phase2Begin = next;
    for (; next < numWorkers && running.load(); ++next) {
      spawnWorkerAt(next);
    }
    std::vector<size_t> spawned;
    for (size_t i = phase2Begin; i < numWorkers; ++i) {
      if (workers[i]->pid.load() > 0) spawned.push_back(i);
    }
    awaitWorkersReady(std::move(spawned), warmupTimeoutMs);

    size_t readyCount = 0;
    for (const auto& w : workers) {
      if (w->isReady.load()) ++readyCount;
    }
    TT_LOG_INFO("[EmbeddingService] Startup finished: {}/{} workers ready",
                readyCount, numWorkers);
  }

  /**
   * Wait for the READY handshake of every listed worker concurrently, via a
   * single poll() over all response pipes. Each worker becomes ready (and
   * gets its dispatch thread) the moment its own sentinel arrives
   */
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

      // 100ms slices keep the shutdown check (`running`) responsive.
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
        if (!(pfds[k].revents & (POLLIN | POLLHUP | POLLERR))) {
          stillPending.push_back(i);
          continue;
        }
        if (pfds[k].revents & POLLIN) {
          const auto msg = pipeReadBinary(workers[i]->readFd.get());
          if (embedding_detail::isReadySentinel(msg)) {
            workers[i]->isReady.store(true);
            TT_LOG_INFO("[EmbeddingService] Worker {} reported ready", i);
            launchDispatchThread(i);
            continue;
          }
          TT_LOG_ERROR(
              "[EmbeddingService] Worker {} exited or sent unexpected data "
              "during warmup",
              i);
        } else {
          // POLLHUP/POLLERR without data: the child died before READY.
          TT_LOG_ERROR(
              "[EmbeddingService] Worker {} closed its pipe during warmup "
              "(process exited)",
              i);
        }
        workers[i]->terminate();
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
      // kill(pid, 0) probes existence without reaping; waitpid stays owned
      // by the dispatch thread (checkAlive) and terminate().
      info.is_alive = info.pid > 0 && kill(info.pid, 0) == 0;
      info.is_ready = w->isReady.load();
      out.push_back(std::move(info));
    }
    return out;
  }

  void stop() {
    if (!running.exchange(false)) return;

    TT_LOG_INFO("[EmbeddingService] Stopping...");
    // The startup thread checks `running` at least every 100ms while waiting
    // on warmups, so this join is quick.
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
    // All consumers are gone; anything still queued would leave its HTTP
    // client hanging forever, so answer every request with an error now.
    drainQueue("Server shutting down");
    {
      std::lock_guard lock(workersMutex);
      workers.clear();
    }
    isReady = false;
    TT_LOG_INFO("[EmbeddingService] Stopped");
  }

  /** Fail every request still waiting in the queue. Callbacks are invoked
   * outside the queue lock: they build HTTP responses and must not serialize
   * against submitters. */
  void drainQueue(const std::string& error) {
    std::queue<std::shared_ptr<PendingRequest>> drained;
    {
      std::lock_guard lock(queueMutex);
      std::swap(drained, requestQueue);
    }
    while (!drained.empty()) {
      auto& p = drained.front();
      domain::EmbeddingResponse err(p->request.task_id);
      err.error = error;
      p->onComplete(std::move(err));
      drained.pop();
    }
  }

  void workerDispatchLoop(size_t workerIdx) {
    auto& worker = workers[workerIdx];
    TT_LOG_INFO("[EmbeddingService] Worker {} dispatch thread started",
                workerIdx);

    uint64_t totalBatches = 0;
    uint64_t totalRequests = 0;
    double totalQueueWaitMs = 0;
    double totalDispatchMs = 0;

    while (worker->running.load() && worker->isReady) {
      std::vector<std::shared_ptr<PendingRequest>> batch;

      auto queueStart = std::chrono::steady_clock::now();
      {
        std::unique_lock lock(queueMutex);
        queueCv.wait_for(lock, std::chrono::milliseconds(100), [this, &worker] {
          return !requestQueue.empty() || !worker->running.load() ||
                 !worker->isReady;
        });

        if (!worker->running.load() || !worker->isReady) break;
        if (requestQueue.empty()) continue;

        if (maxBatchSize > 1 && batchTimeout.count() > 0) {
          while (requestQueue.size() < maxBatchSize) {
            const auto deadline =
                requestQueue.front()->enqueueTime + batchTimeout;
            if (std::chrono::steady_clock::now() >= deadline) break;
            queueCv.wait_until(lock, deadline, [this, &worker] {
              return requestQueue.size() >= maxBatchSize ||
                     requestQueue.empty() || !worker->running.load() ||
                     !worker->isReady;
            });
            if (!worker->running.load() || !worker->isReady) break;
            if (requestQueue.empty()) break;
          }
          if (!worker->running.load() || !worker->isReady) break;
          if (requestQueue.empty()) continue;
        }

        while (batch.size() < maxBatchSize && !requestQueue.empty()) {
          batch.push_back(requestQueue.front());
          requestQueue.pop();
        }
      }
      auto queueEnd = std::chrono::steady_clock::now();
      double queueWaitMs =
          std::chrono::duration<double, std::milli>(queueEnd - queueStart)
              .count();

      if (batch.empty()) continue;

      if (!worker->isReady) {
        failBatch(batch, "Worker died");
        continue;
      }

      totalQueueWaitMs += queueWaitMs;
      totalBatches++;
      totalRequests += batch.size();

      auto dispatchStart = std::chrono::steady_clock::now();
      dispatchBatchToWorker(*worker, batch);
      auto dispatchEnd = std::chrono::steady_clock::now();
      totalDispatchMs +=
          std::chrono::duration<double, std::milli>(dispatchEnd - dispatchStart)
              .count();

      if (totalBatches % 10 == 0) {
        double avgQueue = totalQueueWaitMs / totalBatches;
        double avgDispatch = totalDispatchMs / totalBatches;
        double throughput =
            (totalRequests * 1000.0) / (totalQueueWaitMs + totalDispatchMs);
        TT_LOG_DEBUG(
            "[EmbeddingService] Worker {} batches={} requests={} "
            "avg_queue_wait={}ms avg_dispatch={}ms throughput={} req/s",
            workerIdx, totalBatches, totalRequests, avgQueue, avgDispatch,
            throughput);
      }
    }

    TT_LOG_INFO(
        "[EmbeddingService] Worker {} dispatch thread exiting (isReady={})",
        workerIdx, worker->isReady.load());

    // If this was the last ready worker, the queue has no consumer left and
    // queued callbacks would never fire. During shutdown some workers are
    // still marked ready (stop() only clears `running`), so this drain is
    // skipped there and stop()'s own drain handles the remainder.
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

  void dispatchBatchToWorker(
      WorkerProcess& worker,
      std::vector<std::shared_ptr<PendingRequest>>& batch) {
    if (!worker.isReady.load() || !worker.checkAlive()) {
      failBatch(batch, "Worker not available");
      return;
    }

    Json::Value batchJson(Json::arrayValue);
    for (const auto& p : batch) batchJson.append(p->request.toJson());

    Json::StreamWriterBuilder builder;
    std::string requestStr = Json::writeString(builder, batchJson);

    if (!worker.sendRequest(requestStr)) {
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
        domain::EmbeddingResponse err(pending->request.task_id);
        err.error = "Response not found for task_id";
        pending->onComplete(std::move(err));
      }
    }
  }

  static void failBatch(std::vector<std::shared_ptr<PendingRequest>>& batch,
                        const std::string& error) {
    for (auto& p : batch) {
      domain::EmbeddingResponse err(p->request.task_id);
      err.error = error;
      p->onComplete(std::move(err));
    }
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
