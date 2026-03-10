// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <concepts>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "domain/base_request.hpp"
#include "domain/base_response.hpp"
namespace tt::services {

class QueueFullException : public std::runtime_error {
public:
    QueueFullException() : std::runtime_error("Request queue is full, please retry later") {}
};

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

class IService {
public:
    virtual ~IService() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual bool is_model_ready() const = 0;
    virtual SystemStatus get_system_status() const = 0;
};

template<std::derived_from<domain::BaseRequest> RequestType, std::derived_from<domain::BaseResponse> ResponseType>
class BaseService : public IService {
public:
    virtual ~BaseService() = default;

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

} // namespace tt::services
