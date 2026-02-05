// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <string>
#include <vector>
#include <functional>

#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"

namespace tt::runners {

/**
 * Base class for embedding device runners.
 *
 * Embedding runners generate fixed-size vector representations of text.
 * Unlike LLM runners, they do not stream - they return a single embedding vector.
 */
class BaseEmbeddingRunner {
public:
    explicit BaseEmbeddingRunner(const std::string& device_id)
        : device_id_(device_id) {}

    virtual ~BaseEmbeddingRunner() = default;

    /**
     * Initialize the runner (load model, warmup, etc.)
     * @return true if successful
     */
    virtual bool warmup() = 0;

    /**
     * Clean up resources.
     */
    virtual void close() = 0;

    /**
     * Run embedding inference on a batch of requests.
     * @param requests Batch of embedding requests
     * @return Vector of embedding responses
     */
    virtual std::vector<domain::EmbeddingResponse> run(
        const std::vector<domain::EmbeddingRequest>& requests) = 0;

    /**
     * Get the device ID.
     */
    const std::string& device_id() const { return device_id_; }

protected:
    std::string device_id_;
};

} // namespace tt::runners
