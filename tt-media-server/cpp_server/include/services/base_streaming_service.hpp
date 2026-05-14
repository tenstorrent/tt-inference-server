// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <concepts>
#include <cstdint>
#include <functional>
#include <utility>

#include "domain/base_request.hpp"
#include "services/request_pipeline.hpp"

namespace tt::services {

// Base class for services that produce a stream of chunks for each request
// (e.g. token streams from an LLM). Drives the pipeline
//   preProcess -> produceStream -> streamingPostProcess (per chunk)
// so subclasses focus on the streaming producer.
//
// The chunk type is intentionally not constrained by BaseResponse: streaming
// chunks (such as LLMStreamChunk) are partial deltas, not full responses.
template <std::derived_from<domain::BaseRequest> RequestType,
          typename ChunkType>
class BaseStreamingService : public RequestPipeline<RequestType> {
 public:
  ~BaseStreamingService() override = default;

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

  // Cancel an in-flight streaming request. Default implementation is a
  // no-op; services that support cancellation override it.
  virtual void abortRequest(uint32_t /*taskId*/) {}

 protected:
  virtual void produceStream(
      RequestType request,
      std::function<void(ChunkType&, bool isFinal)> callback) = 0;

  virtual void streamingPostProcess(ChunkType& /*chunk*/) const {}
};

}  // namespace tt::services
