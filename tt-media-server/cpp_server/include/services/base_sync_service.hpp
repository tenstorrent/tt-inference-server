// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#pragma once

#include <concepts>
#include <utility>

#include "domain/base_request.hpp"
#include "domain/base_response.hpp"
#include "services/request_pipeline.hpp"

namespace tt::services {

// Base class for services that produce a single response per request
// (request/response semantics). Drives the pipeline
//   preProcess -> produceResponse -> postProcess
// so subclasses only implement the response producer and any post-processing.
template <std::derived_from<domain::BaseRequest> RequestType,
          std::derived_from<domain::BaseResponse> ResponseType>
class BaseSyncService : public RequestPipeline<RequestType> {
 public:
  ~BaseSyncService() override = default;

  ResponseType submitRequest(RequestType request) {
    this->preProcess(request);
    auto response = produceResponse(std::move(request));
    postProcess(response);
    return response;
  }

 protected:
  virtual ResponseType produceResponse(RequestType request) = 0;
  virtual void postProcess(ResponseType& response) const = 0;
};

}  // namespace tt::services
