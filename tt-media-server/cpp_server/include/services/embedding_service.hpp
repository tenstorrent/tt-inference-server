// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <future>
#include <functional>

#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"
#include "runners/embedding_runner.hpp"

namespace tt::services {

/**
 * Service for handling embedding requests.
 *
 * Uses a multiprocess scheduler with EmbeddingRunner workers.
 * Does not support streaming (embeddings are single-shot).
 */
class EmbeddingService {
public:
    EmbeddingService();
    ~EmbeddingService();

    // Prevent copying
    EmbeddingService(const EmbeddingService&) = delete;
    EmbeddingService& operator=(const EmbeddingService&) = delete;

    /**
     * Start the service and initialize workers.
     */
    void start();

    /**
     * Stop the service and cleanup workers.
     */
    void stop();

    /**
     * Check if the service is ready.
     */
    bool is_ready() const;

    /**
     * Process an embedding request.
     * @param request The embedding request
     * @return Future containing the embedding response
     */
    std::future<domain::EmbeddingResponse> process_request(domain::EmbeddingRequest request);

    /**
     * System status for health checks.
     */
    struct SystemStatus {
        bool model_ready;
        size_t queue_size;
        size_t max_queue_size;
        std::string device;
        size_t num_workers;
    };
    SystemStatus get_system_status() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tt::services
