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
#include <vector>

#include "domain/base_request.hpp"
#include "domain/base_response.hpp"
#include "ipc/warmup_signal_queue.hpp"
#include "utils/logger.hpp"
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

  size_t max_queue_size_ = std::numeric_limits<size_t>::max();

  std::unique_ptr<tt::ipc::IWarmupSignalQueue> warmup_queue_;
  std::thread warmup_listener_thread_;
  std::mutex warmup_mutex_;
  std::condition_variable warmup_cv_;
  std::atomic<bool> warmup_received_{false};
};

}  // namespace tt::services
