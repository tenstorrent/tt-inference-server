// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "runners/runner_interface.hpp"
#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"

namespace tt::runners {

/**
 * Embedding runner that calls Python BGELargeENRunner.
 *
 * Uses Python C API to instantiate and call the BGELargeENRunner class
 * from tt_model_runners/embedding_runner.py.
 */
class EmbeddingRunner : public IRunner {
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
    bool warmup();

    /**
     * Clean up Python objects and optionally finalize interpreter.
     */
    void close();

    /**
     * Run embedding inference by calling runner.run(requests).
     */
    std::vector<domain::EmbeddingResponse> run(
        const std::vector<domain::EmbeddingRequest>& requests);

    // IRunner interface implementation
    void run() override;
    void stop() override;
    const char* runner_type() const override { return "EmbeddingRunner"; }

    /**
     * Get the device ID.
     */
    const std::string& device_id() const { return device_id_; }

private:
    std::string device_id_;
    int visible_device_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tt::runners
