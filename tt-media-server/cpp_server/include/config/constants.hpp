// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "runners/llm_runner/config.hpp"

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

/** Parse MODEL_SERVICE; empty or unknown -> LLM. Expects lowercase input. */
inline ModelService model_service_from_string(const std::string& v) {
    if (v == "embedding") return ModelService::EMBEDDING;
    return ModelService::LLM;
}

enum class RunnerType {
    LLM_TEST,
};

/** String value for env MODEL_RUNNER (e.g. "llm_test"). */
inline std::string to_string(RunnerType r) {
    switch (r) {
        case RunnerType::LLM_TEST:
        default:
            return "llm_test";
    }
}

/** Model type: drives tokenizer strategy + model-specific config. Derived from LLM_DEVICE_BACKEND env var. */
enum class ModelType {
    DEEPSEEK_R1_0528,
    LLAMA_3_1_8B_INSTRUCT,
};


enum class LLMMode {
    REGULAR,
    PREFILL_ONLY,
    DECODE_ONLY,
};

/** String value for env LLM_MODE (e.g. "regular", "prefill", "decode"). */
constexpr std::string_view to_string(LLMMode m) {
    switch (m) {
        case LLMMode::PREFILL_ONLY: return "prefill";
        case LLMMode::DECODE_ONLY:  return "decode";
        case LLMMode::REGULAR:      return "regular";
    }
    return "unknown";
}

/** Parse LLM_MODE; empty or unknown -> REGULAR. Expects lowercase input. */
inline LLMMode llm_mode_from_string(const std::string& v) {
    if (v == "prefill") return LLMMode::PREFILL_ONLY;
    if (v == "decode") return LLMMode::DECODE_ONLY;
    return LLMMode::REGULAR;
}

/** Parse MODEL_RUNNER; unknown -> LLM_TEST. */
inline RunnerType runner_type_from_string(const std::string& /*v*/) {
    return RunnerType::LLM_TEST;
}

/** Map LLM_DEVICE_BACKEND env string to ModelType; "llama" -> LLAMA_3_1_8B_INSTRUCT, else DEEPSEEK_R1_0528. Expects lowercase input. */
inline ModelType model_type_from_device_backend(const std::string& v) {
    if (v == "llama") return ModelType::LLAMA_3_1_8B_INSTRUCT;
    return ModelType::DEEPSEEK_R1_0528;
}

/** Parse SCHEDULING_POLICY; empty or unknown -> PREFILL_FIRST. Expects lowercase input. */
inline llm_engine::SchedulingPolicy scheduling_policy_from_string(const std::string& v) {
    if (v == "max_occupancy") return llm_engine::SchedulingPolicy::MAX_OCCUPANCY;
    return llm_engine::SchedulingPolicy::PREFILL_FIRST;
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
    constexpr const char* TT_PYTHON_PATH = "..";
    constexpr const char* LLM_MODE = "regular";  // "regular", "prefill", "decode"
    constexpr const char* SOCKET_HOST = "localhost";
    constexpr uint16_t SOCKET_PORT = 9000;
    constexpr const char* SCHEDULING_POLICY = "prefill_first";  // "prefill_first" or "max_occupancy"
    constexpr const char* LLM_DEVICE_BACKEND = "mock";  // "mock", "ttrun", "llama"
    constexpr const bool ENABLE_ACCUMULATED_STREAMING = false;
    constexpr size_t MAX_ACCUMULATED_TOKENS = 5;

}

}  // namespace tt::config
