// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <future>
#include <functional>

#include "services/base_service.hpp"
#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"
#include "runners/base_embedding_runner.hpp"

namespace tt::services {

/**
 * Service for handling embedding requests.
 *
 * Uses a multiprocess scheduler with EmbeddingRunner workers.
 * Does not support streaming (embeddings are single-shot).
 */
class EmbeddingService : public BaseService {
public:
    EmbeddingService();
    ~EmbeddingService() override;

    EmbeddingService(const EmbeddingService&) = delete;
    EmbeddingService& operator=(const EmbeddingService&) = delete;

    void start() override;
    void stop() override;
    bool is_model_ready() const override;
    SystemStatus get_system_status() const override;

    /**
     * Process an embedding request.
     * @param request The embedding request
     * @return Future containing the embedding response
     */
    std::future<domain::EmbeddingResponse> process_embedding(domain::EmbeddingRequest request);

protected:
    void process_request(
        domain::CompletionRequest request,
        std::function<void(const domain::StreamingChunkResponse&, bool is_final)> callback
    ) override;

    void pre_process(domain::CompletionRequest& request) const override;
    void post_process(domain::CompletionRequest& request) const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tt::services
