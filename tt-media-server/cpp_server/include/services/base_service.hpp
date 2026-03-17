// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <concepts>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "domain/base_request.hpp"
#include "domain/base_response.hpp"
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

  size_t max_queue_size_ = std::numeric_limits<size_t>::max();
};

}  // namespace tt::services
