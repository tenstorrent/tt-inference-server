// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <atomic>
#include <condition_variable>
#include <concepts>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "domain/base_request.hpp"
#include "domain/base_response.hpp"
#include "ipc/warmup_signal_queue.hpp"
#include "utils/logger.hpp"
#include "worker/single_process_worker.hpp"

namespace tt::services {

class QueueFullException : public std::runtime_error {
 public:
  QueueFullException()
      : std::runtime_error("Request queue is full, please retry later") {}
};

struct WorkerInfo {
  std::string worker_id;
  bool is_ready;
  size_t processed_requests;
};

struct SystemStatus {
  bool model_ready;
  size_t queue_size;
  size_t max_queue_size;
  std::vector<WorkerInfo> worker_info;
};

class IService {
 public:
  virtual ~IService() = default;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual bool isModelReady() const = 0;
  virtual SystemStatus getSystemStatus() const = 0;
};

template <std::derived_from<domain::BaseRequest> RequestType,
          std::derived_from<domain::BaseResponse> ResponseType>
class BaseService : public IService {
 public:
  virtual ~BaseService() = default;

  ResponseType submitRequest(RequestType request) {
    preProcess(request);
    auto response = processRequest(std::move(request));
    postProcess(response);
    return response;
  }

  SystemStatus getSystemStatus() const override {
    SystemStatus status;
    status.model_ready = isModelReady();
    status.queue_size = currentQueueSize();
    status.max_queue_size = max_queue_size_;
    status.worker_info = getWorkerInfo();
    return status;
  }

 protected:
  virtual ResponseType processRequest(RequestType request) = 0;
  virtual void preProcess(RequestType& /*request*/) const {
    if (currentQueueSize() >= max_queue_size_) throw QueueFullException{};
  }
  virtual void postProcess(ResponseType& response) const = 0;
  virtual size_t currentQueueSize() const = 0;

  /** Override to supply worker list for liveness/status (like Python scheduler.get_worker_info()). */
  virtual std::vector<WorkerInfo> getWorkerInfo() const { return {}; }

  /** Override to provide warmup queue (e.g. Boost IPC). Default: no warmup signaling. */
  virtual std::unique_ptr<tt::ipc::IWarmupSignalQueue> createWarmupQueue(
      const std::string& name, size_t capacity) {
    (void)name;
    (void)capacity;
    return nullptr;
  }

  void startWarmupListener(const std::string& name, size_t capacity) {
    warmup_queue_ = createWarmupQueue(name, capacity);
    if (!warmup_queue_) return;
    warmup_received_ = false;
    warmup_listener_thread_ = std::thread([this, capacity]() {
      try {
        for (size_t i = 0; i < capacity; ++i) {
          int workerId = warmup_queue_->receive();
          TT_LOG_INFO("[BaseService] Worker {} warmed up", workerId);
          if (i == 0) {
            warmup_received_ = true;
            warmup_cv_.notify_all();
          }
        }
      } catch (const std::exception& e) {
        TT_LOG_WARN("[BaseService] Warmup listener failed: {} (shutdown?)",
                    e.what());
      } catch (...) {
        TT_LOG_WARN("[BaseService] Warmup listener failed: unknown exception");
      }
    });
  }

  void waitForFirstWarmup() {
    if (!warmup_queue_) return;
    std::unique_lock<std::mutex> lock(warmup_mutex_);
    warmup_cv_.wait(lock, [this]() { return warmup_received_.load(); });
  }

  void stopWarmupListener() {
    if (warmup_queue_) {
      warmup_queue_->remove();
      warmup_queue_.reset();
    }
    if (warmup_listener_thread_.joinable()) {
      warmup_listener_thread_.join();
    }
  }

  virtual worker::WorkerConfig makeWorkerConfig(int workerId);
  virtual void startWorkers();

  size_t max_queue_size_ = std::numeric_limits<size_t>::max();

  std::unique_ptr<tt::ipc::IWarmupSignalQueue> warmup_queue_;
  std::thread warmup_listener_thread_;
  std::mutex warmup_mutex_;
  std::condition_variable warmup_cv_;
  std::atomic<bool> warmup_received_{false};
  std::vector<std::unique_ptr<worker::SingleProcessWorker>> workers_;
  size_t num_workers_ = 0;
};

}  // namespace tt::services

// Template implementation - outside the namespace to avoid pollution
#include <sys/wait.h>
#include <unistd.h>

#include <climits>
#include <cstdio>
#include <cstdlib>

#include "config/settings.hpp"
#include "ipc/boost_ipc_task_queue.hpp"
#include "ipc/queue_manager.hpp"
#include "ipc/shared_memory.hpp"
#include "utils/logger.hpp"

namespace {

[[noreturn]] inline void execWorkerProcessHelper(
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

namespace tt::services {

template <std::derived_from<domain::BaseRequest> RequestType,
          std::derived_from<domain::BaseResponse> ResponseType>
worker::WorkerConfig BaseService<RequestType, ResponseType>::makeWorkerConfig(
    int workerId) {
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

template <std::derived_from<domain::BaseRequest> RequestType,
          std::derived_from<domain::BaseResponse> ResponseType>
void BaseService<RequestType, ResponseType>::startWorkers() {
  for (size_t i = 0; i < num_workers_; i++) {
    auto cfg = makeWorkerConfig(static_cast<int>(i));
    workers_.push_back(std::make_unique<worker::SingleProcessWorker>(cfg));
    auto& worker = workers_[i];

    pid_t pid = fork();

    if (pid < 0) {
      throw std::runtime_error("Failed to fork worker process");
    }
    if (pid == 0) {
      setpgid(0, 0);
      try {
        execWorkerProcessHelper(i, cfg.env_vars);
      } catch (const std::exception& e) {
        TT_LOG_ERROR("[BaseService] Worker {} failed: {}", i, e.what());
        _exit(1);
      }
    }
    setpgid(pid, pid);
    worker->pid = pid;
    TT_LOG_INFO("[BaseService] Spawned worker {} with PID {}", i, pid);
  }
}

// Free function for use by main.cpp worker spawning
inline worker::WorkerConfig makeWorkerConfigForProcess(int workerId) {
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

}  // namespace tt::services
