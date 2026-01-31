// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <functional>

#include "scheduler/scheduler.hpp"
#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"

namespace tt::services {

/**
 * Base service class providing common functionality for all services.
 * Similar to Python's BaseService.
 */
class BaseService {
public:
    BaseService();
    virtual ~BaseService() = default;

    /**
     * Start the service and its workers.
     */
    virtual void start();

    /**
     * Stop the service and cleanup resources.
     */
    virtual void stop();

    /**
     * Check if the model is ready.
     */
    bool is_model_ready() const;

    /**
     * Get system status for monitoring.
     */
    struct SystemStatus {
        bool model_ready;
        size_t queue_size;
        size_t max_queue_size;
        std::string device;
        std::vector<scheduler::Scheduler::WorkerInfo> worker_info;
    };
    SystemStatus get_system_status() const;

    /**
     * Process a non-streaming request.
     */
    std::future<domain::CompletionResponse> process_request(domain::CompletionRequest request);

    /**
     * Process a streaming request.
     * @param request The completion request
     * @param chunk_callback Called for each streaming chunk
     * @param done_callback Called when streaming is complete
     */
    void process_streaming_request(
        domain::CompletionRequest request,
        std::function<void(const domain::StreamingChunkResponse&)> chunk_callback,
        std::function<void()> done_callback
    );

protected:
    /**
     * Pre-process the request before inference.
     * Override in subclass for custom pre-processing.
     */
    virtual domain::CompletionRequest pre_process(domain::CompletionRequest request) {
        return request;
    }

    /**
     * Post-process the response after inference.
     * Override in subclass for custom post-processing.
     */
    virtual domain::CompletionResponse post_process(domain::CompletionResponse response) {
        return response;
    }

    std::shared_ptr<scheduler::Scheduler> scheduler_;
    size_t max_queue_size_ = 10000;
    std::string device_ = "cpu";
};

} // namespace tt::services
