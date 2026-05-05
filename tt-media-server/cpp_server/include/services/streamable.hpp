// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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

  virtual void processStreamingRequest(
      RequestType request,
      std::function<void(const ResponseType&, bool isFinal)> callback) = 0;
};

}  // namespace tt::services
