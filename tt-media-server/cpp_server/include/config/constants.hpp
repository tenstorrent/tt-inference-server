// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstddef>
#include <string>
#include "runners/llm_engine/config.hpp"

namespace tt::config {

/**
 * Constants and enums used by settings, aligned with tt-media-server/config/constants.py.
 */

enum class ModelService {
    LLM,
    EMBEDDING,
};

/** String value for env MODEL_SERVICE (e.g. "llm", "embedding"). */
inline std::string to_string(ModelService s) {
    switch (s) {
        case ModelService::EMBEDDING:
            return "embedding";
        case ModelService::LLM:
        default:
            return "llm";
    }
}

/** Parse MODEL_SERVICE; empty or unknown -> LLM. */
inline ModelService model_service_from_string(const std::string& v) {
    if (v == "embedding") return ModelService::EMBEDDING;
    return ModelService::LLM;
}

enum class RunnerType {
    LLM_TEST,
    TTNN_TEST,
};

/** String value for env MODEL_RUNNER (e.g. "llm_test", "ttnn_test"). */
inline std::string to_string(RunnerType r) {
    switch (r) {
        case RunnerType::TTNN_TEST:
            return "ttnn_test";
        case RunnerType::LLM_TEST:
        default:
            return "llm_test";
    }
}

/** Parse MODEL_RUNNER; unknown -> LLM_TEST. */
inline RunnerType runner_type_from_string(const std::string& v) {
    if (v == "ttnn_test" || v == "TTNN_TEST") return RunnerType::TTNN_TEST;
    return RunnerType::LLM_TEST;
}

/**
 * Default values when the corresponding environment variable is not set or empty.
 * Env overrides these when present.
 */
namespace defaults {
    constexpr const char* DEVICE_IDS = "(0)";
    constexpr const char* MODEL_SERVICE = "llm";
    constexpr size_t MAX_BATCH_SIZE = 1;
    constexpr unsigned MAX_BATCH_DELAY_TIME_MS = 5;
    constexpr const char* MODEL_RUNNER = "llm_test";
    constexpr const char* TT_PYTHON_PATH = "..";
    constexpr llm_engine::Config DEFAULT_LLM_ENGINE_CONFIG = {
        .max_num_batched_tokens = 16384,
        .eos = 0,
        .kvcache_block_size = 128,
        .num_kvcache_blocks = 4000,
    };
}

}  // namespace tt::config
