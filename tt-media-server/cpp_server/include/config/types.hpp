// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace tt::config {

/**
 * Type definitions and enums used throughout the configuration system.
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


/** Model type: drives tokenizer strategy + model-specific config. Derived from LLM_DEVICE_BACKEND env var. */
enum class ModelType {
    DEEPSEEK_R1_0528,
    LLAMA_3_1_8B_INSTRUCT,
};

/** Map LLM_DEVICE_BACKEND env string to ModelType; "llama" -> LLAMA_3_1_8B_INSTRUCT, else DEEPSEEK_R1_0528. Expects lowercase input. */
inline ModelType model_type_from_device_backend(const std::string& v) {
    if (v == "llama") return ModelType::LLAMA_3_1_8B_INSTRUCT;
    return ModelType::DEEPSEEK_R1_0528;
}

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

enum class ModelRunnerType {
    Mock,
    Pipeline,
    Llama
};

enum class SchedulingPolicy {
    PREFILL_FIRST,
    MAX_OCCUPANCY,
};

/** Parse SCHEDULING_POLICY; empty or unknown -> PREFILL_FIRST. Expects lowercase input. */
inline SchedulingPolicy scheduling_policy_from_string(const std::string& v) {
    if (v == "max_occupancy") return SchedulingPolicy::MAX_OCCUPANCY;
    return SchedulingPolicy::PREFILL_FIRST;
}

}  // namespace tt::config
