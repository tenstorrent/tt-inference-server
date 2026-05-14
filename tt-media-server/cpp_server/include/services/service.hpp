// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <stdexcept>
#include <vector>

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
}  // namespace tt::services
