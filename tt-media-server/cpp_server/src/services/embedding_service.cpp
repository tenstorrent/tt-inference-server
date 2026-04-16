// SPDX-License-Identifier: Apache-2.0
#include "utils/id_generator.hpp"
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
#include "services/embedding_codec.hpp"
#include "services/embedding_service.hpp"
#include "utils/logger.hpp"
#include "utils/scoped_fd.hpp"

namespace tt::services {

namespace {

// Length-prefixed pipe write: [len:u32][data].  Returns false on failure.
bool pipeWrite(int fd, const void* data, size_t len) {
  uint32_t header = static_cast<uint32_t>(len);
  if (write(fd, &header, sizeof(header)) != sizeof(header)) return false;
  return write(fd, data, len) == static_cast<ssize_t>(len);
}

// Length-prefixed pipe read.  Returns empty vector on failure.
std::vector<uint8_t> pipeReadBinary(int fd) {
  uint32_t len = 0;
  ssize_t n = read(fd, &len, sizeof(len));
  if (n != sizeof(len) || len > tt::config::defaults::EMBEDDING_MAX_PIPE_BYTES)
    return {};

  std::vector<uint8_t> buf(len);
  size_t total = 0;
  while (total < len) {
    n = read(fd, buf.data() + total, len - total);
    if (n <= 0) return {};
    total += static_cast<size_t>(n);
  }
  return buf;
}

// Length-prefixed pipe read into string.
std::string pipeReadString(int fd) {
  uint32_t len = 0;
  ssize_t n = read(fd, &len, sizeof(len));
  if (n <= 0) return {};

  std::string data(len, '\0');
  size_t total = 0;
  while (total < len) {
    n = read(fd, data.data() + total, len - total);
    if (n <= 0) return {};
    total += static_cast<size_t>(n);
  }
  return data;
}

}  // namespace

struct WorkerProcess {
  int worker_id = -1;
  pid_t pid = -1;
  tt::utils::ScopedFd write_fd;  // parent → child (request pipe write end)
  tt::utils::ScopedFd read_fd;   // child → parent (response pipe read end)
  std::atomic<bool> is_ready{false};
  std::atomic<bool> running{false};
  std::unique_ptr<std::thread> dispatch_thread;

  bool spawn(int workerId,
             std::function<void(int readFd, int writeFd)> childMain) {
    worker_id = workerId;

    int reqRaw[2] = {-1, -1};
    if (pipe(reqRaw) < 0) {
      TT_LOG_ERROR("[EmbeddingService] Failed to create pipes for worker {}",
                   workerId);
      return false;
    }
    tt::utils::ScopedFd reqRead(reqRaw[0]), reqWrite(reqRaw[1]);

    int respRaw[2] = {-1, -1};
    if (pipe(respRaw) < 0) {
      TT_LOG_ERROR("[EmbeddingService] Failed to create pipes for worker {}",
                   workerId);
      return false;  // reqRead + reqWrite auto-close
    }
    tt::utils::ScopedFd respRead(respRaw[0]), respWrite(respRaw[1]);

    pid_t child = fork();
    if (child < 0) {
      TT_LOG_ERROR("[EmbeddingService] Failed to fork worker {}", workerId);
      return false;  // all 4 FDs auto-close
    }

    if (child == 0) {
      // Child: close parent ends, run child main.
      reqWrite.reset();
      respRead.reset();
      childMain(reqRead.release(), respWrite.release());
      _exit(0);  // childMain is [[noreturn]], but just in case
    }

    // Parent: close child ends, transfer ownership to members.
    reqRead.reset();
    respWrite.reset();
    pid = child;
    write_fd = std::move(reqWrite);
    read_fd = std::move(respRead);
    is_ready.store(true);
    running.store(true);

    TT_LOG_INFO(
        "[EmbeddingService] Spawned worker {} with PID {} "
        "(TT_VISIBLE_DEVICES={}) write_fd={} read_fd={}",
        workerId, pid, tt::config::visibleDevicesForWorker(workerId),
        write_fd.get(), read_fd.get());
    return true;
  }

  bool checkAlive() {
    if (pid <= 0) return false;
    int status;
    pid_t result = waitpid(pid, &status, WNOHANG);
    if (result != pid) return true;

    if (WIFEXITED(status)) {
      TT_LOG_ERROR("[EmbeddingService] Worker {} exited with code {}",
                   worker_id, WEXITSTATUS(status));
    } else if (WIFSIGNALED(status)) {
      TT_LOG_ERROR("[EmbeddingService] Worker {} killed by signal {}",
                   worker_id, WTERMSIG(status));
    }
    is_ready.store(false);
    return false;
  }

  bool sendRequest(const std::string& json) {
    if (!pipeWrite(write_fd.get(), json.data(), json.size())) {
      TT_LOG_ERROR("[EmbeddingService] Worker {} pipe write failed: {}",
                   worker_id, strerror(errno));
      is_ready.store(false);
      return false;
    }
    return true;
  }

  std::vector<uint8_t> receiveResponse() {
    auto buf = pipeReadBinary(read_fd.get());
    if (buf.empty()) {
      TT_LOG_ERROR("[EmbeddingService] Worker {} response read failed",
                   worker_id);
      is_ready.store(false);
    }
    return buf;
  }

  void terminate() {
    if (pid > 0) {
      kill(pid, SIGTERM);
      waitpid(pid, nullptr, 0);
      TT_LOG_INFO("[EmbeddingService] Worker {} terminated", worker_id);
    }
    write_fd.reset();
    read_fd.reset();
  }
};

struct EmbeddingService::Impl {
  struct PendingRequest {
    domain::EmbeddingRequest request;
    std::promise<domain::EmbeddingResponse> promise;
    explicit PendingRequest(domain::EmbeddingRequest req)
        : request(std::move(req)) {}
  };

  std::vector<std::unique_ptr<WorkerProcess>> workers_;
  size_t num_workers_ = 3;

  TRACY_LOCKABLE(std::mutex, queue_mutex_);
  std::queue<std::shared_ptr<PendingRequest>> request_queue_;
  std::condition_variable_any queue_cv_;

  std::atomic<bool> running_{false};
  std::atomic<bool> is_ready_{false};

  size_t max_batch_size_ = 1;
  std::chrono::milliseconds batch_timeout_{5};
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

  [[noreturn]] static void workerProcessMain(int workerId, int readFd,
                                             int writeFd) {
    size_t wid = static_cast<size_t>(workerId);
    std::string visibleDevices = tt::config::visibleDevicesForWorker(wid);
    setenv("TT_VISIBLE_DEVICES", visibleDevices.c_str(), 1);

    TT_LOG_INFO("[Worker {}] Started (PID {}, TT_VISIBLE_DEVICES={})", workerId,
                getpid(), visibleDevices);

    runners::EmbeddingRunner runner(visibleDevices, workerId);
    if (!runner.warmup()) {
      TT_LOG_ERROR("[Worker {}] Warmup failed!", workerId);
      _exit(1);
    }
    TT_LOG_INFO("[Worker {}] Ready", workerId);

    while (true) {
      std::string requestJson = pipeReadString(readFd);
      if (requestJson.empty()) break;

      Json::Value reqJson;
      Json::CharReaderBuilder builder;
      std::istringstream iss(requestJson);
      std::string errors;
      if (!Json::parseFromStream(builder, iss, &reqJson, &errors)) {
        TT_LOG_ERROR("[Worker {}] Failed to parse request: {}", workerId,
                     errors);
        continue;
      }

      auto taskIdFromJson = [](const Json::Value& j) -> uint32_t {
        return (j.isMember("task_id") && j["task_id"].isUInt())
                   ? j["task_id"].asUInt()
                   : tt::utils::TaskIDGenerator::generate();
      };

      std::vector<domain::EmbeddingRequest> batch;
      if (reqJson.isArray()) {
        for (const auto& item : reqJson)
          batch.push_back(
              domain::EmbeddingRequest::fromJson(item, taskIdFromJson(item)));
      } else {
        batch.push_back(domain::EmbeddingRequest::fromJson(
            reqJson, taskIdFromJson(reqJson)));
      }

      TT_LOG_INFO("[Worker {}] Processing batch of {} requests", workerId,
                  batch.size());

      auto responses = runner.run(batch);
      auto buf = embedding_codec::encodeResponses(batch, responses);

      if (!pipeWrite(writeFd, buf.data(), buf.size())) {
        TT_LOG_ERROR("[Worker {}] Failed to write response", workerId);
      }
    }

    runner.close();
    _exit(0);
  }

  void start() {
    if (running_.exchange(true)) return;

    TT_LOG_INFO("[EmbeddingService] Starting with {} worker processes",
                num_workers_);
    workers_.reserve(num_workers_);

    for (size_t i = 0; i < num_workers_; ++i) {
      auto w = std::make_unique<WorkerProcess>();
      int wid = static_cast<int>(i);
      if (!w->spawn(
              wid, [wid](int rd, int wr) { workerProcessMain(wid, rd, wr); })) {
        continue;
      }
      workers_.push_back(std::move(w));
    }

    for (size_t i = 0; i < workers_.size(); ++i) {
      workers_[i]->dispatch_thread =
          std::make_unique<std::thread>(&Impl::workerDispatchLoop, this, i);
    }

    is_ready_ = true;
    TT_LOG_INFO("[EmbeddingService] All {} workers started", workers_.size());
  }

  void stop() {
    if (!running_.exchange(false)) return;

    TT_LOG_INFO("[EmbeddingService] Stopping...");
    queue_cv_.notify_all();

    for (auto& w : workers_) w->running = false;
    queue_cv_.notify_all();

    size_t batchSize = tt::config::maxInFlightCount();
    std::string batchStr = std::to_string(batchSize);
    setenv("MAX_BATCH_SIZE", batchStr.c_str(), 1);

    for (auto& w : workers_) {
      if (w->dispatch_thread && w->dispatch_thread->joinable())
        w->dispatch_thread->join();
      w->terminate();
    }
    workers_.clear();
    is_ready_ = false;
    TT_LOG_INFO("[EmbeddingService] Stopped");
  }

  void workerDispatchLoop(size_t workerIdx) {
    auto& worker = workers_[workerIdx];
    TT_LOG_INFO("[EmbeddingService] Worker {} dispatch thread started",
                workerIdx);

    uint64_t totalBatches = 0;
    uint64_t totalRequests = 0;
    double totalQueueWaitMs = 0;
    double totalDispatchMs = 0;

    while (worker->running.load() && worker->is_ready) {
      std::vector<std::shared_ptr<PendingRequest>> batch;

      auto queueStart = std::chrono::steady_clock::now();
      {
        std::unique_lock lock(queue_mutex_);
        queue_cv_.wait_for(
            lock, std::chrono::milliseconds(100), [this, &worker] {
              return !request_queue_.empty() || !worker->running.load() ||
                     !worker->is_ready;
            });

        if (!worker->running.load() || !worker->is_ready) break;
        if (request_queue_.empty()) continue;

        while (batch.size() < max_batch_size_ && !request_queue_.empty()) {
          batch.push_back(request_queue_.front());
          request_queue_.pop();
        }
      }
      auto queueEnd = std::chrono::steady_clock::now();
      double queueWaitMs =
          std::chrono::duration<double, std::milli>(queueEnd - queueStart)
              .count();

      if (batch.empty()) continue;

      if (!worker->is_ready) {
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
        "[EmbeddingService] Worker {} dispatch thread exiting (is_ready={})",
        workerIdx, worker->is_ready.load());
  }

  void dispatchBatchToWorker(
      WorkerProcess& worker,
      std::vector<std::shared_ptr<PendingRequest>>& batch) {
    if (!worker.is_ready.load() || !worker.checkAlive()) {
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
        pending->promise.set_value(std::move(it->second));
      } else {
        domain::EmbeddingResponse err(pending->request.task_id);
        err.error = "Response not found for task_id";
        pending->promise.set_value(std::move(err));
      }
    }
  }

  static void failBatch(std::vector<std::shared_ptr<PendingRequest>>& batch,
                        const std::string& error) {
    for (auto& p : batch) {
      domain::EmbeddingResponse err(p->request.task_id);
      err.error = error;
      p->promise.set_value(std::move(err));
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
    queue_cv_.notify_all();

    return future;
  }
};

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

void EmbeddingService::postProcess(domain::EmbeddingResponse&) const {}

domain::EmbeddingResponse EmbeddingService::processRequest(
    domain::EmbeddingRequest request) {
  auto future = impl_->submitRequest(std::move(request));
  return future.get();
}

}  // namespace tt::services
