// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstddef>
#include <string>

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

/** Runner type from MODEL_RUNNER: LLM_TEST (in-process stub) or LLAMA_RUNNER (Python tt_model_runners.llama_runner). */
enum class RunnerType {
    LLM_TEST,
    LLAMA_RUNNER,
};

/** MODEL_RUNNER string for env (llm_test -> LLM_TEST, llama_runner -> LLAMA_RUNNER). */
inline std::string to_string(RunnerType r) {
    return r == RunnerType::LLAMA_RUNNER ? "llama_runner" : "llm_test";
}

/** Parse MODEL_RUNNER; unknown -> LLM_TEST. */
inline RunnerType runner_type_from_string(const std::string& v) {
    return v == "llama_runner" ? RunnerType::LLAMA_RUNNER : RunnerType::LLM_TEST;
}

/**
 * Default values when the corresponding environment variable is not set or empty.
 * Env overrides these when present.
 */
namespace defaults {
    constexpr const char* DEVICE_IDS = "(0)";
    constexpr const char* MODEL_SERVICE = "llm";
    constexpr const char* MODEL_RUNNER = "llm_test";
    constexpr size_t MAX_BATCH_SIZE = 1;
    constexpr unsigned MAX_BATCH_DELAY_TIME_MS = 5;
    constexpr const char* TT_PYTHON_PATH = "..";
}

}  // namespace tt::config
