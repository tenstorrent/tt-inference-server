// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <limits>
#include <stdexcept>
#include <vector>

#include "domain/base_request.hpp"
#include "domain/base_response.hpp"
#include "worker/worker_info.hpp"

namespace tt::services {

class QueueFullException : public std::runtime_error {
 public:
  QueueFullException()
      : std::runtime_error("Request queue is full, please retry later") {}
};

struct SystemStatus {
  bool modelReady;
  size_t queueSize;
  size_t maxQueueSize;
  std::vector<tt::worker::WorkerInfo> workerInfo;
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
    status.modelReady = isModelReady();
    status.queueSize = currentQueueSize();
    status.maxQueueSize = maxQueueSize;
    status.workerInfo = getWorkerInfo();
    return status;
  }

  bool isModelReady() const override { return false; }

 protected:
  virtual ResponseType processRequest(RequestType request) = 0;
  virtual void preProcess(RequestType& /*request*/) const {
    if (currentQueueSize() >= maxQueueSize) throw QueueFullException{};
  }
  virtual void postProcess(ResponseType& response) const = 0;
  virtual size_t currentQueueSize() const = 0;

  virtual std::vector<tt::worker::WorkerInfo> getWorkerInfo() const {
    return {};
  }

  size_t maxQueueSize = std::numeric_limits<size_t>::max();
};

}  // namespace tt::services
