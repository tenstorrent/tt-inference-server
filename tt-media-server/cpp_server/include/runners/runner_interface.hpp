// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <string>

namespace tt::runners {

/**
 * Common interface for all runners (LLM, Embedding, etc.).
 * Provides basic lifecycle management for inference runners.
 */
class IRunner {
public:
    virtual ~IRunner() = default;

    /**
     * Start the runner and begin processing.
     * This method should run the main inference loop.
     */
    virtual void run() = 0;

    /**
     * Stop the runner gracefully.
     */
    virtual void stop() = 0;

    /**
     * Get the runner type for identification.
     */
    virtual const char* runner_type() const = 0;
};

} // namespace tt::runners
