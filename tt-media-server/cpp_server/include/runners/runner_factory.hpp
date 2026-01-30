// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <memory>
#include <string>
#include <cstdlib>
#include <iostream>

#include "runners/base_device_runner.hpp"
#include "runners/llm_test_runner.hpp"

#ifdef ENABLE_TTNN
#include "runners/ttnn_test_runner.hpp"
#endif

namespace tt::runners {

/**
 * Runner factory that creates the appropriate runner based on environment variable.
 *
 * Environment variable: TT_RUNNER_TYPE
 *   - "llm_test" (default): Use LLMTestRunner (pure CPU, 120k tokens/sec)
 *   - "ttnn_test": Use TTNNTestRunner (TTNN device I/O) - requires ENABLE_TTNN build flag
 */
class RunnerFactory {
public:
    enum class RunnerType {
        LLM_TEST,
        TTNN_TEST
    };

    /**
     * Get the runner type from environment variable TT_RUNNER_TYPE.
     */
    static RunnerType get_runner_type() {
        const char* runner_env = std::getenv("TT_RUNNER_TYPE");
        if (runner_env == nullptr) {
            return RunnerType::LLM_TEST;  // Default
        }

        std::string runner_type(runner_env);
        if (runner_type == "ttnn_test" || runner_type == "TTNN_TEST") {
#ifdef ENABLE_TTNN
            return RunnerType::TTNN_TEST;
#else
            std::cerr << "[RunnerFactory] WARNING: TTNN runner requested but not compiled with ENABLE_TTNN. "
                      << "Rebuild with -DENABLE_TTNN=ON to enable TTNN support. "
                      << "Falling back to LLM test runner." << std::endl;
            return RunnerType::LLM_TEST;
#endif
        }

        return RunnerType::LLM_TEST;
    }

    /**
     * Get a human-readable name for the runner type.
     */
    static std::string get_runner_name(RunnerType type) {
        switch (type) {
            case RunnerType::TTNN_TEST:
                return "TTNNTestRunner (Device I/O)";
            case RunnerType::LLM_TEST:
            default:
                return "LLMTestRunner (120k tokens/sec)";
        }
    }

    /**
     * Create a runner instance based on the environment variable.
     */
    static std::unique_ptr<BaseDeviceRunner> create(const std::string& device_id) {
        RunnerType type = get_runner_type();

        std::cout << "[RunnerFactory] Creating runner: " << get_runner_name(type)
                  << " for device " << device_id << std::endl;

        switch (type) {
#ifdef ENABLE_TTNN
            case RunnerType::TTNN_TEST:
                return std::make_unique<TTNNTestRunner>(device_id);
#endif
            case RunnerType::LLM_TEST:
            default:
                return std::make_unique<LLMTestRunner>(device_id);
        }
    }

    /**
     * Get a factory function for use with Scheduler.
     */
    static std::function<std::unique_ptr<BaseDeviceRunner>(const std::string&)> get_factory() {
        return [](const std::string& device_id) {
            return create(device_id);
        };
    }
};

} // namespace tt::runners
