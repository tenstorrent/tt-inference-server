// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <string>

namespace tt::config {

/**
 * Constants and enums used by settings, aligned with tt-media-server/config/constants.py.
 */

enum class ModelService {
    LLM,
    EMBEDDING,
};

/** String value for env TT_MODEL_SERVICE (e.g. "llm", "embedding"). */
inline std::string to_string(ModelService s) {
    switch (s) {
        case ModelService::EMBEDDING:
            return "embedding";
        case ModelService::LLM:
        default:
            return "llm";
    }
}

/** Parse TT_MODEL_SERVICE; empty or unknown -> LLM. */
inline ModelService model_service_from_string(const std::string& v) {
    if (v == "embedding") return ModelService::EMBEDDING;
    return ModelService::LLM;
}

enum class RunnerType {
    LLM_TEST,
    TTNN_TEST,
};

/** String value for env TT_RUNNER_TYPE (e.g. "llm_test", "ttnn_test"). */
inline std::string to_string(RunnerType r) {
    switch (r) {
        case RunnerType::TTNN_TEST:
            return "ttnn_test";
        case RunnerType::LLM_TEST:
        default:
            return "llm_test";
    }
}

/** Parse TT_RUNNER_TYPE; unknown -> LLM_TEST. */
inline RunnerType runner_type_from_string(const std::string& v) {
    if (v == "ttnn_test" || v == "TTNN_TEST") return RunnerType::TTNN_TEST;
    return RunnerType::LLM_TEST;
}

/** Environment variable names used when setting worker process env (e.g. in embedding_service). */
namespace env_keys {
    constexpr const char* TT_VISIBLE_DEVICES = "TT_VISIBLE_DEVICES";
    constexpr const char* TT_DEVICE_ID = "TT_DEVICE_ID";
    constexpr const char* TT_WORKER_ID = "TT_WORKER_ID";
}

}  // namespace tt::config
