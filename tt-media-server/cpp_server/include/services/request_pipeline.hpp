// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "domain/base_request.hpp"
#include "domain/base_response.hpp"
#include "runtime/worker/worker_info.hpp"

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
  virtual std::string runnerInUse() const { return ""; }
};

/** Shared queue back-pressure and system status for all request pipelines. */
template <std::derived_from<domain::BaseRequest> RequestType>
class RequestPipeline : public IService {
 public:
  virtual ~RequestPipeline() = default;

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
  virtual void preProcess(RequestType& /*request*/) const {
    enforceQueueCapacity();
  }

  void enforceQueueCapacity() const {
    if (currentQueueSize() >= maxQueueSize) {
      throw QueueFullException{};
    }
  }

  /** Override when the service has its own queue; default is no back-pressure.
   */
  virtual size_t currentQueueSize() const { return 0; }

  virtual std::vector<tt::worker::WorkerInfo> getWorkerInfo() const {
    return {};
  }

  size_t maxQueueSize = std::numeric_limits<size_t>::max();
};

/** Single-request services: one response per submitRequest. */
template <std::derived_from<domain::BaseRequest> RequestType,
          std::derived_from<domain::BaseResponse> ResponseType>
class BaseSyncService : public RequestPipeline<RequestType> {
 public:
  virtual ~BaseSyncService() = default;

  ResponseType submitRequest(RequestType request) {
    this->preProcess(request);
    return produceResponse(std::move(request));
  }

 protected:
  virtual ResponseType produceResponse(RequestType request) = 0;
};

/** Streaming services: many chunks per request via callback. */
template <std::derived_from<domain::BaseRequest> RequestType,
          std::derived_from<domain::BaseResponse> ChunkType>
class BaseStreamingService : public RequestPipeline<RequestType> {
 public:
  virtual ~BaseStreamingService() = default;

  void submitStreamingRequest(
      RequestType& request,
      std::function<void(const ChunkType&, bool isFinal)> callback,
      bool skipPreProcess = false) {
    if (!skipPreProcess) {
      this->preProcess(request);
    }
    produceStream(std::move(request), [this, cb = std::move(callback)](
                                          ChunkType& chunk, bool isFinal) {
      streamingPostProcess(chunk);
      cb(chunk, isFinal);
    });
  }

  virtual void abortRequest(uint32_t /*taskId*/) {}

 protected:
  virtual void produceStream(
      RequestType request,
      std::function<void(ChunkType&, bool isFinal)> callback) = 0;

  virtual void streamingPostProcess(ChunkType& /*chunk*/) const {}
};

}  // namespace tt::services
