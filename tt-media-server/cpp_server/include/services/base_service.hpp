// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <concepts>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "domain/base_request.hpp"
#include "domain/base_response.hpp"
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
    // worker_info is empty by default, services can populate it if needed
    return status;
  }

 protected:
  virtual ResponseType processRequest(RequestType request) = 0;
  virtual void preProcess(RequestType& /*request*/) const {
    if (currentQueueSize() >= max_queue_size_) throw QueueFullException{};
  }
  virtual void postProcess(ResponseType& response) const = 0;
  virtual size_t currentQueueSize() const = 0;

  // Worker management - override makeWorkerConfig to customize worker setup
  virtual worker::WorkerConfig makeWorkerConfig(int workerId);

  // Default worker startup implementation
  virtual void startWorkers();

  size_t max_queue_size_ = std::numeric_limits<size_t>::max();
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
