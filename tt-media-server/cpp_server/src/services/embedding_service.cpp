// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "services/embedding_service.hpp"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "config/defaults.hpp"
#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "runners/embedding_runner.hpp"
#include "utils/logger.hpp"

namespace tt::services {

/**
 * Implementation using fork-based multiprocessing.
 *
 * Each worker process:
 * - Has its own TT_VISIBLE_DEVICES environment variable
 * - Runs an EmbeddingRunner instance
 * - Communicates via pipes
 */
struct EmbeddingService::Impl {
  // Forward declare PendingRequest first
  struct PendingRequest {
    domain::EmbeddingRequest request;
    std::promise<domain::EmbeddingResponse> promise;

    explicit PendingRequest(domain::EmbeddingRequest req)
        : request(std::move(req)) {}
  };

  // Worker process info
  struct WorkerProcess {
    pid_t pid = -1;
    int worker_id = -1;
    int request_pipe[2] = {-1, -1};   // Parent writes, child reads
    int response_pipe[2] = {-1, -1};  // Child writes, parent reads
    std::atomic<bool> is_ready{false};

    // Per-worker dispatch thread (pulls from shared queue)
    std::unique_ptr<std::thread> dispatch_thread;
    std::atomic<bool> running{false};
  };

  std::vector<std::unique_ptr<WorkerProcess>> workers_;
  size_t num_workers_ = 3;  // Default 3 workers for devices 1, 2, 3

  // Shared request queue (all worker dispatch threads pull from this)
  TRACY_LOCKABLE(std::mutex, queue_mutex_);
  std::queue<std::shared_ptr<PendingRequest>> request_queue_;
  std::condition_variable_any queue_cv_;

  std::atomic<bool> running_{false};
  std::atomic<bool> is_ready_{false};

  // Batching configuration
  // NOTE: max_batch_size_ must match Python's MAX_BATCH_SIZE setting
  // Set TT_BATCH_SIZE and MAX_BATCH_SIZE env vars to the same value
  size_t max_batch_size_ =
      1;  // Max requests per batch (default 1 = no batching)
  std::chrono::milliseconds batch_timeout_{5};  // Max wait time to fill batch
  size_t max_queue_size_ = tt::config::defaults::MAX_QUEUE_SIZE;

  Impl() {
    num_workers_ = tt::config::numWorkers();
    max_batch_size_ = tt::config::maxInFlightCount();
    batch_timeout_ = std::chrono::milliseconds(tt::config::batchTimeoutMs());
    max_queue_size_ = tt::config::maxQueueSize();
    TT_LOG_INFO(
        "[EmbeddingService] Initialized with {} workers, batch_size={}, "
        "batch_timeout={}ms",
        num_workers_, max_batch_size_, batch_timeout_.count());
  }

  ~Impl() { stop(); }

  void start() {
    if (running_.exchange(true)) {
      return;
    }

    TT_LOG_INFO("[EmbeddingService] Starting with {} worker processes",
                num_workers_);

    workers_.reserve(num_workers_);

    // First, fork all worker processes
    for (size_t i = 0; i < num_workers_; ++i) {
      auto worker = std::make_unique<WorkerProcess>();
      worker->worker_id = static_cast<int>(i);

      // Create pipes
      if (pipe(worker->request_pipe) < 0 || pipe(worker->response_pipe) < 0) {
        TT_LOG_ERROR("[EmbeddingService] Failed to create pipes for worker {}",
                     i);
        continue;
      }

      pid_t pid = fork();

      if (pid < 0) {
        TT_LOG_ERROR("[EmbeddingService] Failed to fork worker {}", i);
        continue;
      } else if (pid == 0) {
        // Child process
        workerProcessMain(static_cast<int>(i), worker->request_pipe,
                          worker->response_pipe);
        // Never returns
      } else {
        // Parent process
        worker->pid = pid;

        // Close unused pipe ends
        close(worker->request_pipe[0]);   // Close read end of request pipe
        close(worker->response_pipe[1]);  // Close write end of response pipe

        worker->is_ready.store(true);
        worker->running.store(true);

        TT_LOG_INFO(
            "[EmbeddingService] Spawned worker {} with PID {} "
            "(TT_VISIBLE_DEVICES={}) request_pipe[1]={} response_pipe[0]={}",
            i, pid, tt::config::visibleDevicesForWorker(i),
            worker->request_pipe[1], worker->response_pipe[0]);

        workers_.push_back(std::move(worker));
      }
    }

    // Now start per-worker dispatch threads (after all workers are in the
    // vector)
    for (size_t i = 0; i < workers_.size(); ++i) {
      workers_[i]->dispatch_thread =
          std::make_unique<std::thread>(&Impl::workerDispatchLoop, this, i);
    }

    is_ready_ = true;
    TT_LOG_INFO("[EmbeddingService] All {} workers started", workers_.size());
  }

  void stop() {
    if (!running_.exchange(false)) {
      return;
    }

    TT_LOG_INFO("[EmbeddingService] Stopping...");

    // Wake up all dispatch threads
    queue_cv_.notify_all();

    // Stop and join per-worker dispatch threads
    for (auto& worker : workers_) {
      worker->running = false;
    }
    queue_cv_.notify_all();  // Wake them up again after setting running=false

    size_t batchSize = tt::config::maxInFlightCount();
    std::string batchStr = std::to_string(batchSize);
    setenv("MAX_BATCH_SIZE", batchStr.c_str(), 1);

    for (auto& worker : workers_) {
      if (worker->dispatch_thread && worker->dispatch_thread->joinable()) {
        worker->dispatch_thread->join();
      }

      if (worker->pid > 0) {
        kill(worker->pid, SIGTERM);
        waitpid(worker->pid, nullptr, 0);
        TT_LOG_INFO("[EmbeddingService] Worker {} terminated",
                    worker->worker_id);
      }

      // Close pipes
      if (worker->request_pipe[1] >= 0) close(worker->request_pipe[1]);
      if (worker->response_pipe[0] >= 0) close(worker->response_pipe[0]);
    }

    workers_.clear();
    is_ready_ = false;

    TT_LOG_INFO("[EmbeddingService] Stopped");
  }

  [[noreturn]] void workerProcessMain(int workerId, int requestPipe[2],
                                      int responsePipe[2]) {
    // Save our FDs first, before closing anything
    int readFd = requestPipe[0];
    int writeFd = responsePipe[1];

    // Close unused pipe ends for THIS worker
    close(requestPipe[1]);   // Close write end of request pipe
    close(responsePipe[0]);  // Close read end of response pipe

    size_t wid = static_cast<size_t>(workerId);
    std::string visibleDevices = tt::config::visibleDevicesForWorker(wid);
    setenv("TT_VISIBLE_DEVICES", visibleDevices.c_str(), 1);

    TT_LOG_INFO("[Worker {}] Started with PID {}", workerId, getpid());
    TT_LOG_INFO("[Worker {}] TT_VISIBLE_DEVICES={}", workerId, visibleDevices);
    TT_LOG_INFO("[Worker {}] read_fd={}, write_fd={}", workerId, readFd,
                writeFd);

    runners::EmbeddingRunner runner(visibleDevices, static_cast<int>(wid));

    // Warmup
    if (!runner.warmup()) {
      TT_LOG_ERROR("[Worker {}] Warmup failed!", workerId);
      _exit(1);
    }

    TT_LOG_INFO("[Worker {}] Ready", workerId);

    // Process requests (supports batching)
    while (true) {
      // Read request length
      uint32_t requestLen = 0;
      ssize_t n = read(readFd, &requestLen, sizeof(requestLen));
      if (n <= 0) {
        TT_LOG_ERROR("[Worker {}] Pipe closed or read error (n={})", workerId,
                     n);
        break;  // Pipe closed or error
      }

      TT_LOG_DEBUG("[Worker {}] Reading request of {} bytes", workerId,
                   requestLen);

      // Read request JSON - loop until all bytes are read
      std::string requestJson(requestLen, '\0');
      size_t totalRead = 0;
      while (totalRead < requestLen) {
        n = read(readFd, requestJson.data() + totalRead,
                 requestLen - totalRead);
        if (n <= 0) {
          TT_LOG_ERROR("[Worker {}] Read error at {}/{}", workerId, totalRead,
                       requestLen);
          break;
        }
        totalRead += n;
      }
      if (totalRead != requestLen) {
        TT_LOG_ERROR("[Worker {}] Failed to read full request", workerId);
        continue;
      }

      // Parse request (can be array for batch or object for single)
      Json::Value reqJson;
      Json::CharReaderBuilder builder;
      std::istringstream iss(requestJson);
      std::string errors;
      if (!Json::parseFromStream(builder, iss, &reqJson, &errors)) {
        TT_LOG_ERROR("[Worker {}] Failed to parse request: {}", workerId,
                     errors);
        continue;
      }

      // Build batch of requests
      std::vector<domain::EmbeddingRequest> batch;
      auto taskIdFromJson = [](const Json::Value& j) -> domain::TaskID {
        if (j.isMember("task_id") && j["task_id"].isUInt()) {
          return j["task_id"].asUInt();
        }
        return domain::TaskIDGenerator::generate();
      };
      if (reqJson.isArray()) {
        for (const auto& item : reqJson) {
          batch.push_back(
              domain::EmbeddingRequest::fromJson(item, taskIdFromJson(item)));
        }
      } else {
        batch.push_back(domain::EmbeddingRequest::fromJson(
            reqJson, taskIdFromJson(reqJson)));
      }

      TT_LOG_INFO("[Worker {}] Processing batch of {} requests", workerId,
                  batch.size());

      // Run inference on batch
      auto responses = runner.run(batch);

      // Build binary response format:
      // [num_responses: uint32_t]
      // For each response:
      //   [task_id_len: uint32_t][task_id: chars]
      //   [has_error: uint8_t]
      //   If has_error:
      //     [error_len: uint32_t][error: chars]
      //   Else:
      //     [embedding_dim: uint32_t][embedding: floats]
      //     [total_tokens: int32_t]
      //     [model_len: uint32_t][model: chars]

      std::vector<uint8_t> responseBuffer;
      responseBuffer.reserve(batch.size() * (4 + 32 + 1 + 4 + 1024 * 4 + 4 + 4 +
                                             32));  // Estimate size

      // Helper to append data
      auto appendUint32 = [&responseBuffer](uint32_t val) {
        responseBuffer.insert(responseBuffer.end(),
                              reinterpret_cast<uint8_t*>(&val),
                              reinterpret_cast<uint8_t*>(&val) + sizeof(val));
      };
      auto appendInt32 = [&responseBuffer](int32_t val) {
        responseBuffer.insert(responseBuffer.end(),
                              reinterpret_cast<uint8_t*>(&val),
                              reinterpret_cast<uint8_t*>(&val) + sizeof(val));
      };
      auto appendString = [&responseBuffer,
                           &appendUint32](const std::string& s) {
        appendUint32(static_cast<uint32_t>(s.size()));
        responseBuffer.insert(responseBuffer.end(), s.begin(), s.end());
      };
      auto appendFloats = [&responseBuffer,
                           &appendUint32](const std::vector<float>& floats) {
        appendUint32(static_cast<uint32_t>(floats.size()));
        const uint8_t* data = reinterpret_cast<const uint8_t*>(floats.data());
        responseBuffer.insert(responseBuffer.end(), data,
                              data + floats.size() * sizeof(float));
      };

      appendUint32(static_cast<uint32_t>(batch.size()));

      for (size_t i = 0; i < batch.size(); ++i) {
        appendUint32(batch[i].task_id);

        if (i < responses.size() && responses[i].error.empty()) {
          responseBuffer.push_back(0);  // has_error = false
          appendFloats(responses[i].embedding);
          appendInt32(responses[i].total_tokens);
          appendString(responses[i].model);
        } else {
          responseBuffer.push_back(1);  // has_error = true
          std::string error = (i < responses.size())
                                  ? responses[i].error
                                  : "No response from runner";
          appendString(error);
        }
      }

      uint32_t responseLen = static_cast<uint32_t>(responseBuffer.size());
      TT_LOG_DEBUG("[Worker {}] Sending binary response of {} bytes", workerId,
                   responseLen);

      ssize_t w1 = write(writeFd, &responseLen, sizeof(responseLen));
      ssize_t w2 = write(writeFd, responseBuffer.data(), responseLen);

      if (w1 != sizeof(responseLen) ||
          w2 != static_cast<ssize_t>(responseLen)) {
        TT_LOG_ERROR("[Worker {}] Failed to write response: w1={} w2={}",
                     workerId, w1, w2);
      }
    }

    runner.close();
    _exit(0);
  }

  // Per-worker dispatch loop: pulls from shared queue, batches, and sends to
  // worker process
  void workerDispatchLoop(size_t workerIdx) {
    auto& worker = workers_[workerIdx];

    TT_LOG_INFO("[EmbeddingService] Worker {} dispatch thread started",
                workerIdx);

    // Performance counters
    uint64_t totalBatches = 0;
    uint64_t totalRequests = 0;
    double totalQueueWaitMs = 0;
    double totalDispatchMs = 0;

    while (worker->running.load() && worker->is_ready) {
      std::vector<std::shared_ptr<PendingRequest>> batch;

      auto queueStart = std::chrono::steady_clock::now();
      {
        std::unique_lock lock(queue_mutex_);

        // Wait for at least one request (with timeout to check running flag)
        auto waitResult = queue_cv_.wait_for(
            lock, std::chrono::milliseconds(100), [this, &worker] {
              return !request_queue_.empty() || !worker->running.load() ||
                     !worker->is_ready;
            });

        // Check if we should exit
        if (!worker->running.load() || !worker->is_ready) {
          break;
        }

        if (!waitResult || request_queue_.empty()) {
          continue;  // Timeout or spurious wakeup, check again
        }

        // Grab up to max_batch_size_ requests that are already in the queue
        // Don't wait for more - release lock quickly so other workers can grab
        // work
        while (batch.size() < max_batch_size_ && !request_queue_.empty()) {
          batch.push_back(request_queue_.front());
          request_queue_.pop();
        }
      }
      auto queueEnd = std::chrono::steady_clock::now();
      double queueWaitMs =
          std::chrono::duration<double, std::milli>(queueEnd - queueStart)
              .count();
      // Lock released here - other workers can now grab from queue

      if (!batch.empty() && worker->is_ready) {
        totalQueueWaitMs += queueWaitMs;
        totalBatches++;
        totalRequests += batch.size();

        auto dispatchStart = std::chrono::steady_clock::now();
        dispatchBatchToWorker(*worker, batch);
        auto dispatchEnd = std::chrono::steady_clock::now();
        double dispatchMs = std::chrono::duration<double, std::milli>(
                                dispatchEnd - dispatchStart)
                                .count();
        totalDispatchMs += dispatchMs;

        // Log every 10 batches
        if (totalBatches % 10 == 0) {
          double avgQueueWait = totalQueueWaitMs / totalBatches;
          double avgDispatch = totalDispatchMs / totalBatches;
          double throughput =
              (totalRequests * 1000.0) / (totalQueueWaitMs + totalDispatchMs);
          TT_LOG_DEBUG(
              "[EmbeddingService] Worker {} batches={} requests={} "
              "avg_queue_wait={}ms avg_dispatch={}ms throughput={} req/s",
              workerIdx, totalBatches, totalRequests, avgQueueWait, avgDispatch,
              throughput);
        }
      } else if (!batch.empty()) {
        // Worker died while we were grabbing work, put it back or error out
        TT_LOG_ERROR("[EmbeddingService] Worker {} died, failing {} requests",
                     workerIdx, batch.size());
        for (auto& pending : batch) {
          domain::EmbeddingResponse errorResp(pending->request.task_id);
          errorResp.error = "Worker died";
          pending->promise.set_value(errorResp);
        }
      }
    }

    TT_LOG_INFO(
        "[EmbeddingService] Worker {} dispatch thread exiting (is_ready={})",
        workerIdx, worker->is_ready.load());
  }

  // Send batch to specific worker and wait for response (blocking)
  void dispatchBatchToWorker(
      WorkerProcess& worker,
      std::vector<std::shared_ptr<PendingRequest>>& batch) {
    auto t0 = std::chrono::steady_clock::now();

    if (!worker.is_ready.load() || worker.pid <= 0) {
      for (auto& pending : batch) {
        domain::EmbeddingResponse errorResp(pending->request.task_id);
        errorResp.error = "Worker not available";
        pending->promise.set_value(errorResp);
      }
      return;
    }

    // Check if worker process is still alive
    int status;
    pid_t result = waitpid(worker.pid, &status, WNOHANG);
    if (result == worker.pid) {
      // Worker has exited
      if (WIFEXITED(status)) {
        TT_LOG_ERROR("[EmbeddingService] Worker {} exited with code {}",
                     worker.worker_id, WEXITSTATUS(status));
      } else if (WIFSIGNALED(status)) {
        TT_LOG_ERROR("[EmbeddingService] Worker {} killed by signal {}",
                     worker.worker_id, WTERMSIG(status));
      }
      worker.is_ready.store(false);
      for (auto& pending : batch) {
        domain::EmbeddingResponse errorResp(pending->request.task_id);
        errorResp.error = "Worker process died";
        pending->promise.set_value(errorResp);
      }
      return;
    }

    auto t1 = std::chrono::steady_clock::now();

    // Build batch request JSON
    Json::Value batchJson(Json::arrayValue);
    for (const auto& pending : batch) {
      batchJson.append(pending->request.toJson());
    }

    Json::StreamWriterBuilder builder;
    std::string requestStr = Json::writeString(builder, batchJson);
    uint32_t requestLen = static_cast<uint32_t>(requestStr.size());

    auto t2 = std::chrono::steady_clock::now();

    // Send to worker
    ssize_t written =
        write(worker.request_pipe[1], &requestLen, sizeof(requestLen));
    if (written != sizeof(requestLen)) {
      TT_LOG_ERROR("[EmbeddingService] Worker {} failed to write length: {}",
                   worker.worker_id, strerror(errno));
      worker.is_ready.store(false);  // Mark worker as dead
      for (auto& pending : batch) {
        domain::EmbeddingResponse errorResp(pending->request.task_id);
        errorResp.error = "Worker pipe broken - worker crashed";
        pending->promise.set_value(errorResp);
      }
      return;
    }
    written = write(worker.request_pipe[1], requestStr.data(), requestLen);
    if (written != static_cast<ssize_t>(requestLen)) {
      TT_LOG_ERROR("[EmbeddingService] Worker {} failed to write data: {}",
                   worker.worker_id, strerror(errno));
      worker.is_ready.store(false);  // Mark worker as dead
      for (auto& pending : batch) {
        domain::EmbeddingResponse errorResp(pending->request.task_id);
        errorResp.error = "Worker pipe broken - worker crashed";
        pending->promise.set_value(errorResp);
      }
      return;
    }

    auto t3 = std::chrono::steady_clock::now();

    // Read response
    uint32_t responseLen = 0;
    ssize_t n =
        read(worker.response_pipe[0], &responseLen, sizeof(responseLen));

    TT_LOG_DEBUG(
        "[EmbeddingService] Worker {} got binary response length: {} (read {} "
        "bytes)",
        worker.worker_id, responseLen, n);

    // Sanity check - response should be reasonable size (< 100MB)
    if (n != sizeof(responseLen) || responseLen > 100 * 1024 * 1024) {
      TT_LOG_ERROR(
          "[EmbeddingService] Worker {} invalid response length {} - pipe "
          "corrupted?",
          worker.worker_id, responseLen);
      worker.is_ready.store(false);
      for (auto& pending : batch) {
        domain::EmbeddingResponse errorResp(pending->request.task_id);
        errorResp.error = "Failed to read response from worker";
        pending->promise.set_value(errorResp);
      }
      return;
    }

    std::vector<uint8_t> responseBuffer(responseLen);
    size_t totalRead = 0;
    while (totalRead < responseLen) {
      n = read(worker.response_pipe[0], responseBuffer.data() + totalRead,
               responseLen - totalRead);
      if (n <= 0) {
        TT_LOG_ERROR("[EmbeddingService] Worker {} read error at {}/{} bytes",
                     worker.worker_id, totalRead, responseLen);
        break;
      }
      totalRead += n;
    }
    if (totalRead != responseLen) {
      TT_LOG_ERROR("[EmbeddingService] Worker {} incomplete read: {}/{} bytes",
                   worker.worker_id, totalRead, responseLen);
      for (auto& pending : batch) {
        domain::EmbeddingResponse errorResp(pending->request.task_id);
        errorResp.error = "Failed to read full response from worker";
        pending->promise.set_value(errorResp);
      }
      return;
    }

    auto t4 = std::chrono::steady_clock::now();

    // Parse binary response
    // Format:
    // [num_responses: uint32_t]
    // For each response:
    //   [task_id_len: uint32_t][task_id: chars]
    //   [has_error: uint8_t]
    //   If has_error:
    //     [error_len: uint32_t][error: chars]
    //   Else:
    //     [embedding_dim: uint32_t][embedding: floats]
    //     [total_tokens: int32_t]
    //     [model_len: uint32_t][model: chars]

    size_t offset = 0;
    auto readUint32 = [&responseBuffer, &offset]() -> uint32_t {
      uint32_t val;
      std::memcpy(&val, responseBuffer.data() + offset, sizeof(val));
      offset += sizeof(val);
      return val;
    };
    auto readInt32 = [&responseBuffer, &offset]() -> int32_t {
      int32_t val;
      std::memcpy(&val, responseBuffer.data() + offset, sizeof(val));
      offset += sizeof(val);
      return val;
    };
    auto readString = [&responseBuffer, &offset, &readUint32]() -> std::string {
      uint32_t len = readUint32();
      std::string s(
          reinterpret_cast<const char*>(responseBuffer.data() + offset), len);
      offset += len;
      return s;
    };
    auto readFloats = [&responseBuffer, &offset,
                       &readUint32]() -> std::vector<float> {
      uint32_t count = readUint32();
      std::vector<float> floats(count);
      std::memcpy(floats.data(), responseBuffer.data() + offset,
                  count * sizeof(float));
      offset += count * sizeof(float);
      return floats;
    };

    // Parse responses into a map by task_id
    std::unordered_map<std::string, domain::EmbeddingResponse> responseMap;
    uint32_t numResponses = readUint32();
    responseMap.reserve(numResponses);

    for (uint32_t i = 0; i < numResponses && offset < responseBuffer.size();
         ++i) {
      domain::EmbeddingResponse resp{readUint32()};
      uint8_t hasError = responseBuffer[offset++];

      if (hasError) {
        resp.error = readString();
      } else {
        resp.embedding = readFloats();
        resp.total_tokens = readInt32();
        resp.model = readString();
      }

      auto key = std::to_string(resp.task_id);
      responseMap.insert_or_assign(std::move(key), std::move(resp));
    }

    auto t5 = std::chrono::steady_clock::now();

    // Log timing breakdown
    double checkMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double buildJsonMs =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    double writePipeMs =
        std::chrono::duration<double, std::milli>(t3 - t2).count();
    double waitWorkerMs =
        std::chrono::duration<double, std::milli>(t4 - t3).count();
    double parseBinaryMs =
        std::chrono::duration<double, std::milli>(t5 - t4).count();
    double totalMs = std::chrono::duration<double, std::milli>(t5 - t0).count();
    double overheadMs = totalMs - waitWorkerMs;

    // Always log timing for every batch
    TT_LOG_DEBUG(
        "[EmbeddingService] Worker {} batch={} check={}ms build={}ms "
        "write={}ms wait={}ms parse={}ms overhead={}ms total={}ms",
        worker.worker_id, batch.size(), checkMs, buildJsonMs, writePipeMs,
        waitWorkerMs, parseBinaryMs, overheadMs, totalMs);

    // Match responses to requests by task_id
    for (auto& pending : batch) {
      auto it = responseMap.find(std::to_string(pending->request.task_id));
      if (it != responseMap.end()) {
        pending->promise.set_value(std::move(it->second));
      } else {
        domain::EmbeddingResponse errorResp(pending->request.task_id);
        errorResp.error = "Response not found for task_id";
        pending->promise.set_value(std::move(errorResp));
      }
    }
  }

  std::future<domain::EmbeddingResponse> submitRequest(
      domain::EmbeddingRequest request) {
    auto pending = std::make_shared<PendingRequest>(std::move(request));
    auto future = pending->promise.get_future();

    {
      std::lock_guard lock(queue_mutex_);
      request_queue_.push(pending);
    }
    queue_cv_.notify_all();  // Wake all workers so they can compete for work

    return future;
  }
};

// Public interface

EmbeddingService::EmbeddingService() : impl_(std::make_unique<Impl>()) {
  max_queue_size_ = impl_->max_queue_size_;
}

EmbeddingService::~EmbeddingService() = default;

void EmbeddingService::start() { impl_->start(); }

void EmbeddingService::stop() { impl_->stop(); }

bool EmbeddingService::isModelReady() const { return impl_->is_ready_.load(); }

size_t EmbeddingService::currentQueueSize() const {
  std::lock_guard lock(impl_->queue_mutex_);
  return impl_->request_queue_.size();
}

void EmbeddingService::postProcess(domain::EmbeddingResponse&) const {
  // no-op
}

domain::EmbeddingResponse EmbeddingService::processRequest(
    domain::EmbeddingRequest request) {
  auto future = impl_->submitRequest(std::move(request));
  return future.get();
}

}  // namespace tt::services
