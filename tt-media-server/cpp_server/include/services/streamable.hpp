// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <concepts>
#include <functional>

#include "domain/base_request.hpp"
#include "domain/base_response.hpp"

namespace tt::services {

template<std::derived_from<domain::BaseRequest> RequestType, std::derived_from<domain::BaseResponse> ResponseType>
class Streamable {
public:
    virtual ~Streamable() = default;

    void submit_streaming_request(
        RequestType& request,
        std::function<void(const ResponseType&, bool is_final)> callback
    ) {
        pre_process(request);
        process_streaming_request(std::move(request),
            [this, cb = std::move(callback)](ResponseType& response, bool is_final) {
                streaming_post_process(response);
                cb(response, is_final);
            });
    }

protected:
    virtual void process_streaming_request(
        RequestType request,
        std::function<void(ResponseType&, bool is_final)> callback
    ) = 0;

    virtual void pre_process(RequestType& request) const = 0;
    virtual void streaming_post_process(ResponseType& response) const = 0;
};

} // namespace tt::services
