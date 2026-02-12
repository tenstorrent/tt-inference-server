// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <functional>
#include <future>
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

    /**
     * Get system status for monitoring.
     */
    virtual SystemStatus get_system_status() const = 0;

    /**
     * Process a non-streaming request.
     */
    virtual std::future<domain::CompletionResponse> process_request(domain::CompletionRequest request) = 0;

    /**
     * Process a streaming request.
     * @param request The completion request
     * @param chunk_callback Called for each streaming chunk
     * @param done_callback Called when streaming is complete
     */
    virtual void process_streaming_request(
        domain::CompletionRequest request,
        std::function<void(const domain::StreamingChunkResponse&)> chunk_callback,
        std::function<void()> done_callback
    ) = 0;

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
};

} // namespace tt::services
