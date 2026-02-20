// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <functional>
#include <string>
#include <vector>

namespace tt::services {

struct WorkerInfo {
    std::string worker_id;
    bool is_ready;
    size_t processed_requests;
};

struct SystemStatus {
    bool model_ready;
    size_t queue_size;
    size_t max_queue_size;
    std::string device;
    std::vector<WorkerInfo> worker_info;
};

/**
 * Base service providing lifecycle management and a sync submission path.
 * pre_process runs before dispatch, post_process runs on the returned response.
 */
template<typename RequestType, typename ResponseType>
class BaseService {
public:
    virtual ~BaseService() = default;

    virtual void start() = 0;
    virtual void stop() = 0;
    virtual bool is_model_ready() const = 0;
    virtual SystemStatus get_system_status() const = 0;

    ResponseType submit_request(RequestType request) {
        pre_process(request);
        auto response = process_request(std::move(request));
        post_process(response);
        return response;
    }

protected:
    virtual ResponseType process_request(RequestType request) = 0;
    virtual void pre_process(RequestType& request) const = 0;
    virtual void post_process(ResponseType& response) const = 0;
};

/**
 * Mixin for services that support streaming responses.
 * post_process is applied to each chunk before forwarding to the caller.
 */
template<typename RequestType, typename ResponseType>
class Streamable {
public:
    virtual ~Streamable() = default;

    void submit_streaming_request(
        RequestType request,
        std::function<void(const ResponseType&, bool is_final)> callback
    ) {
        streaming_pre_process(request);
        process_streaming_request(std::move(request),
            [this, cb = std::move(callback)](ResponseType& response, bool is_final) {
                streaming_post_process(response);
                cb(response, is_final);
            });
    }

protected:
    virtual void process_streaming_request(
        RequestType request,
        std::function<void(const ResponseType&, bool is_final)> callback
    ) = 0;

    virtual void streaming_pre_process(RequestType& request) const = 0;
    virtual void streaming_post_process(ResponseType& response) const = 0;
};

} // namespace tt::services
