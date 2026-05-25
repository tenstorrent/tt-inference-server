// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <concepts>
#include <limits>
#include <vector>

#include "domain/base_request.hpp"
#include "services/service.hpp"
#include "runtime/worker/worker_info.hpp"

namespace tt::services {

// Shared request-handling scaffolding for any service that accepts requests
// of a given type. Holds queue-capacity policy, worker-info reporting, and
// the SystemStatus assembly used by every concrete service.
//
// Sync (BaseSyncService) and streaming (BaseStreamingService) services
// derive from this so they share one queue/status contract regardless of
// how the response is delivered.
template <std::derived_from<domain::BaseRequest> RequestType>
class RequestPipeline : public IService {
 public:
  ~RequestPipeline() override = default;

  // Validate/normalize the request before it enters the service. Default
  // behavior enforces queue capacity; concrete services override to add
  // request-type-specific checks but should still call into the base.
  virtual void preProcess(RequestType& /*request*/) const {
    enforceQueueCapacity();
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
  void enforceQueueCapacity() const {
    if (currentQueueSize() >= maxQueueSize) throw QueueFullException{};
  }

  virtual size_t currentQueueSize() const = 0;

  virtual std::vector<tt::worker::WorkerInfo> getWorkerInfo() const {
    return {};
  }

  size_t maxQueueSize = std::numeric_limits<size_t>::max();
};

}  // namespace tt::services
