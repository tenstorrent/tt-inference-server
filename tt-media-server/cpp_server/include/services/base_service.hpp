// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"

namespace tt::services {

/**
 * Abstract base service class defining the interface for all completion services.
 * Concrete implementations (e.g. LLMService) provide scheduling, worker management,
 * and request dispatching.
 */
class BaseService {
public:
    virtual ~BaseService() = default;

    /**
     * Start the service and its workers.
     */
    virtual void start() = 0;

    /**
     * Stop the service and cleanup resources.
     */
    virtual void stop() = 0;

    /**
     * Check if the model is ready.
     */
    virtual bool is_model_ready() const = 0;

    /**
     * Worker info for monitoring.
     */
    struct WorkerInfo {
        std::string worker_id;
        bool is_ready;
        size_t processed_requests;
    };

    /**
     * System status for monitoring.
     */
    struct SystemStatus {
        bool model_ready;
        size_t queue_size;
        size_t max_queue_size;
        std::string device;
        std::vector<WorkerInfo> worker_info;
    };

    virtual SystemStatus get_system_status() const = 0;

    virtual void process_request(
        domain::CompletionRequest request,
        std::function<void(const domain::StreamingChunkResponse&)> chunk_callback,
        std::function<void()> done_callback
    ) = 0;

protected:
    /**
     * Pre-process request before dispatching (e.g. tokenize string prompt).
     */
    virtual void pre_process(domain::CompletionRequest& request) = 0;
};

} // namespace tt::services
