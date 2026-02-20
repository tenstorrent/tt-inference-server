// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <string>
#include <functional>
#include <coroutine>

#include "domain/completion_request.hpp"
#include "domain/completion_response.hpp"

namespace tt::runners {

/**
 * Base class for all device runners.
 * Similar to Python's BaseDeviceRunner.
 */
class BaseDeviceRunner {
public:
    explicit BaseDeviceRunner(const std::string& device_id)
        : device_id_(device_id) {}

    virtual ~BaseDeviceRunner() = default;

    /**
     * Warm up the runner (load model, etc.).
     * @return true if warmup successful
     */
    virtual bool warmup() = 0;

    /**
     * Run inference on a batch of requests (non-streaming).
     */
    virtual std::vector<domain::CompletionResponse> run(
        const std::vector<domain::CompletionRequest>& requests) = 0;

    /**
     * Close the device and cleanup resources.
     */
    virtual void close() {}

    const std::string& device_id() const { return device_id_; }

protected:
    std::string device_id_;
};

} // namespace tt::runners
