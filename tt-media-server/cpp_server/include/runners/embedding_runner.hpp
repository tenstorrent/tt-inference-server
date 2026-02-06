// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>

#include "runners/base_embedding_runner.hpp"

namespace tt::runners {

/**
 * Embedding runner that calls Python BGELargeENRunner.
 *
 * Uses Python C API to instantiate and call the BGELargeENRunner class
 * from tt_model_runners/embedding_runner.py.
 */
class EmbeddingRunner : public BaseEmbeddingRunner {
public:
    /** @param device_id e.g. "device_0". @param visible_device TT device index (1-based) for logging. */
    EmbeddingRunner(const std::string& device_id, int visible_device = 0);
    ~EmbeddingRunner() override;

    // Prevent copying
    EmbeddingRunner(const EmbeddingRunner&) = delete;
    EmbeddingRunner& operator=(const EmbeddingRunner&) = delete;

    /**
     * Initialize Python, import modules, create BGELargeENRunner instance,
     * and call warmup().
     */
    bool warmup() override;

    /**
     * Clean up Python objects and optionally finalize interpreter.
     */
    void close() override;

    /**
     * Run embedding inference by calling runner.run(requests).
     */
    std::vector<domain::EmbeddingResponse> run(
        const std::vector<domain::EmbeddingRequest>& requests) override;

private:
    int visible_device_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tt::runners
