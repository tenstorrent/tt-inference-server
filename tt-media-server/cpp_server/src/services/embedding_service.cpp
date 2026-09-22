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
#include <deque>
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
  /**
   * One queued request plus its completion callback.
   */
  struct PendingRequest {
    domain::EmbeddingRequest request;

    std::function<void(domain::EmbeddingResponse&&)> onComplete;

    // Arrival time; anchors the batch-fill deadline in collectBatch.
    std::chrono::steady_clock::time_point enqueueTime;

    PendingRequest(domain::EmbeddingRequest req,
                   std::function<void(domain::EmbeddingResponse&&)> complete)
        : request(std::move(req)),
          onComplete(std::move(complete)),
          enqueueTime(std::chrono::steady_clock::now()) {}
  };

  /**
   * Complete one pending request with an error response (the onComplete
   * exactly-once contract: this must be the request's only completion).
   */
  static void completeWithError(PendingRequest& p, const std::string& error) {
    domain::EmbeddingResponse err(p.request.task_id);
    err.error = error;
    p.onComplete(std::move(err));
  }

  std::vector<std::unique_ptr<WorkerProcess>> workers;

  /// Pipeline depth per worker: one batch on the device plus one being
  /// tokenized in the worker child (double buffering).
  static constexpr size_t kMaxBatchesInFlight = 2;

  /**
   * Batches sent to a worker whose responses have not arrived yet, in send
   * order. The dispatch (sender) thread pushes after a successful write and
   * blocks while the deque is full; the receive thread pops as responses
   * arrive. Responses come back in send order because the worker child
   * processes its pipe serially.
   */
  struct InFlightState {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::vector<std::shared_ptr<PendingRequest>>> batches;
  };
  std::vector<std::unique_ptr<InFlightState>> inFlight;

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

  /** Allocate the worker table and hand spawning/warmup to the startup
   * thread, so start() returns immediately and health probes get answers
   * while the model loads. */
  void start() {
    if (running.exchange(true)) return;

    TT_LOG_INFO("[EmbeddingService] Starting with {} worker processes",
                numWorkers);
    {
      std::lock_guard lock(workersMutex);
      workers.reserve(numWorkers);
      inFlight.reserve(numWorkers);
      for (size_t i = 0; i < numWorkers; ++i) {
        auto w = std::make_unique<WorkerProcess>();
        w->workerId = static_cast<int>(i);
        workers.push_back(std::move(w));
        inFlight.push_back(std::make_unique<InFlightState>());
      }
    }

    startupThread = std::make_unique<std::thread>(&Impl::runStartup, this);
  }

  /**
   * Bring up one worker alone and wait for its READY handshake before
   * spawning the rest. The first warmup on a cold volume generates the shared
   * tensor cache; once one worker has written it, the remaining workers warm
   * up in parallel safely.
   */
  void runStartup() {
    const unsigned warmupTimeoutMs = tt::config::embeddingWarmupTimeoutMs();
    const size_t next = warmupCacheLeader(warmupTimeoutMs);
    spawnAndAwaitRemaining(next, warmupTimeoutMs);
    logStartupSummary();
  }

  /**
   * Phase 1: warm up a single worker with exclusive cache access, so it can
   * populate the shared tensor cache without racing. If it fails, try the
   * next worker alone. Flips isReady on the first success. Returns the index
   * of the first worker not yet spawned.
   */
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

  /** Phase 2: the tensor cache is warm; the remaining workers spawn at once
   * and warm up concurrently. */
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

  /**
   * Handle one poll() event for a warming-up worker. Returns true when the
   * worker is resolved — it became ready (dispatch thread launched) or it
   * died and was terminated; false while its handshake is still pending.
   */
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
      // POLLHUP/POLLERR without data: the child died before READY.
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
    workers[idx]->receiveThread =
        std::make_unique<std::thread>(&Impl::workerReceiveLoop, this, idx);
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
    for (auto& s : inFlight) {
      if (s) s->cv.notify_all();
    }

    for (auto& w : workers) {
      if (w->dispatchThread && w->dispatchThread->joinable())
        w->dispatchThread->join();
    }
    // terminate() ends the child, which EOFs the response pipe and unblocks
    // a receive thread parked in receiveResponse().
    for (auto& w : workers) {
      w->terminate();
      if (w->receiveThread && w->receiveThread->joinable())
        w->receiveThread->join();
    }
    // All consumers are gone; anything still queued would leave its HTTP
    // client hanging forever, so answer every request with an error now.
    drainQueue("Server shutting down");
    {
      std::lock_guard lock(workersMutex);
      workers.clear();
      inFlight.clear();
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
      completeWithError(*drained.front(), error);
      drained.pop();
    }
  }

  /** Rolling dispatch-loop statistics; logs a summary every 10 batches. */
  struct DispatchStats {
    uint64_t batches = 0;
    uint64_t requests = 0;
    double queueWaitMs = 0;
    double sendMs = 0;

    void record(size_t workerIdx, size_t batchSize, double qMs, double sMs) {
      queueWaitMs += qMs;
      batches++;
      requests += batchSize;
      sendMs += sMs;

      if (batches % 10 == 0) {
        double avgQueue = queueWaitMs / batches;
        double avgSend = sendMs / batches;
        TT_LOG_DEBUG(
            "[EmbeddingService] Worker {} batches={} requests={} "
            "avg_queue_wait={}ms avg_send={}ms",
            workerIdx, batches, requests, avgQueue, avgSend);
      }
    }
  };

  /**
   * Block until requests arrive or the worker must exit, then take up to
   * maxBatchSize requests off the queue. When the queue is non-empty but a
   * full batch has not formed, wait (mutex released) until batchTimeout past
   * the OLDEST queued request's arrival
   */
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

  /**
   * Dispatch-thread exit path: if this was the last ready worker, the queue
   * has no consumer left and queued callbacks would never fire. During
   * shutdown workers are still marked ready (stop() only clears `running`),
   * so the drain is skipped here and stop()'s own drain handles the
   * remainder.
   */
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

  /**
   * Sender half of the per-worker pipeline: collect a batch, write it to the
   * request pipe, and record it as in flight. Runs up to kMaxBatchesInFlight
   * batches ahead of the receive thread, so the worker child can tokenize
   * batch N+1 while batch N occupies the device.
   */
  void workerDispatchLoop(size_t workerIdx) {
    auto& worker = workers[workerIdx];
    auto& inflight = *inFlight[workerIdx];
    TT_LOG_INFO("[EmbeddingService] Worker {} dispatch thread started",
                workerIdx);

    DispatchStats stats;

    while (worker->running.load() && worker->isReady) {
      {
        std::unique_lock lock(inflight.mutex);
        inflight.cv.wait(lock, [&] {
          return inflight.batches.size() < kMaxBatchesInFlight ||
                 !worker->running.load() || !worker->isReady;
        });
      }
      if (!worker->running.load() || !worker->isReady) break;

      const auto queueStart = std::chrono::steady_clock::now();
      auto batch = collectBatch(*worker);
      const auto queueEnd = std::chrono::steady_clock::now();

      if (batch.empty()) continue;

      if (!worker->isReady) {
        failBatch(batch, "Worker died");
        continue;
      }

      const size_t batchSize = batch.size();
      const auto sendStart = std::chrono::steady_clock::now();
      if (!worker->checkAlive() ||
          !worker->sendRequest(encodeBatchJson(batch))) {
        failBatch(batch, "Worker not available");
        continue;
      }
      const auto sendEnd = std::chrono::steady_clock::now();

      {
        std::lock_guard lock(inflight.mutex);
        inflight.batches.push_back(std::move(batch));
      }
      inflight.cv.notify_all();

      stats.record(
          workerIdx, batchSize,
          std::chrono::duration<double, std::milli>(queueEnd - queueStart)
              .count(),
          std::chrono::duration<double, std::milli>(sendEnd - sendStart)
              .count());
    }

    TT_LOG_INFO(
        "[EmbeddingService] Worker {} dispatch thread exiting (isReady={})",
        workerIdx, worker->isReady.load());

    // Wake the receive thread so it notices isReady/running went false.
    inflight.cv.notify_all();

    drainIfLastWorker();
  }

  /**
   * Receiver half of the per-worker pipeline: wait for a batch to be in
   * flight, block on the response pipe, and complete the oldest in-flight
   * batch with what arrived. Exits once the worker is stopped or dead and
   * every in-flight batch has been answered or failed.
   */
  void workerReceiveLoop(size_t workerIdx) {
    auto& worker = workers[workerIdx];
    auto& inflight = *inFlight[workerIdx];

    while (true) {
      {
        std::unique_lock lock(inflight.mutex);
        inflight.cv.wait(lock, [&] {
          return !inflight.batches.empty() || !worker->running.load() ||
                 !worker->isReady;
        });
        if (inflight.batches.empty()) break;
      }

      auto responseBuf = worker->receiveResponse();

      std::vector<std::shared_ptr<PendingRequest>> batch;
      {
        std::lock_guard lock(inflight.mutex);
        batch = std::move(inflight.batches.front());
        inflight.batches.pop_front();
      }
      inflight.cv.notify_all();

      if (responseBuf.empty()) {
        // receiveResponse cleared isReady; the next loop iteration drains
        // any remaining in-flight batch and then exits.
        failBatch(batch, "Failed to read response from worker");
        continue;
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

    TT_LOG_INFO("[EmbeddingService] Worker {} receive thread exiting",
                workerIdx);
  }

  /** JSON-encode a batch as the array payload the worker's serve loop
   * expects. */
  static std::string encodeBatchJson(
      const std::vector<std::shared_ptr<PendingRequest>>& batch) {
    Json::Value batchJson(Json::arrayValue);
    for (const auto& p : batch) batchJson.append(p->request.toJson());
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, batchJson);
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
