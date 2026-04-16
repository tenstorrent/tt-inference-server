// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <concepts>
#include <functional>

#include "domain/base_request.hpp"
#include "domain/base_response.hpp"

namespace tt::services {

template <std::derived_from<domain::BaseRequest> RequestType,
          std::derived_from<domain::BaseResponse> ResponseType>
class Streamable {
 public:
  virtual ~Streamable() = default;

  void submitStreamingRequest(
      RequestType& request,
      std::function<void(const ResponseType&, bool isFinal)> callback) {
    processStreamingRequest(
        std::move(request),
        [this, cb = std::move(callback)](ResponseType& response, bool isFinal) {
          streamingPostProcess(response);
          cb(response, isFinal);
        });
  }

 protected:
  virtual void processStreamingRequest(
      RequestType request,
      std::function<void(ResponseType&, bool isFinal)> callback) = 0;

  virtual void preProcess(RequestType& request) const = 0;
  virtual void streamingPostProcess(ResponseType& response) const = 0;
};

}  // namespace tt::services
