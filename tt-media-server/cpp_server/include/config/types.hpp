// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tt::config {

/**
 * Type definitions and enums used throughout the configuration system.
 */

enum class ModelService {
  LLM,
  EMBEDDING,
  IMAGE,
};

/** String value for env MODEL_SERVICE. */
inline std::string toString(ModelService s) {
  switch (s) {
    case ModelService::EMBEDDING:
      return "embedding";
    case ModelService::IMAGE:
      return "image";
    case ModelService::LLM:
    default:
      return "llm";
  }
}

/** Parse MODEL_SERVICE; empty or unknown -> LLM. Expects lowercase input. */
inline ModelService modelServiceFromString(const std::string& v) {
  if (v == "embedding") return ModelService::EMBEDDING;
  if (v == "image") return ModelService::IMAGE;
  return ModelService::LLM;
}

/** Model type: drives tokenizer strategy + model-specific config. Derived from
 * LLM_DEVICE_BACKEND env var. */
enum class ModelType {
  DEEPSEEK_R1_0528,
  LLAMA_3_1_8B_INSTRUCT,
  KIMI_K2_6,
};

/** Map lowercase `LLM_DEVICE_BACKEND` values to `ModelType`:
 * "llama" -> `LLAMA_3_1_8B_INSTRUCT`,
 * "kimi" -> `KIMI_K2_6`,
 * otherwise -> `DEEPSEEK_R1_0528`. */
inline ModelType modelTypeFromDeviceBackend(const std::string& v) {
  if (v == "llama")
    return ModelType::LLAMA_3_1_8B_INSTRUCT;
  else if (v == "kimi")
    return ModelType::KIMI_K2_6;
  return ModelType::DEEPSEEK_R1_0528;
}

enum class LLMMode {
  REGULAR,
  PREFILL_ONLY,
  DECODE_ONLY,
};

/** String value for env LLM_MODE (e.g. "regular", "prefill", "decode"). */
constexpr std::string_view toString(LLMMode m) {
  switch (m) {
    case LLMMode::PREFILL_ONLY:
      return "prefill";
    case LLMMode::DECODE_ONLY:
      return "decode";
    case LLMMode::REGULAR:
      return "regular";
  }
  return "unknown";
}

/** Parse LLM_MODE; empty or unknown -> REGULAR. Expects lowercase input. */
inline LLMMode llmModeFromString(const std::string& v) {
  if (v == "prefill") return LLMMode::PREFILL_ONLY;
  if (v == "decode") return LLMMode::DECODE_ONLY;
  return LLMMode::REGULAR;
}

enum class ModelRunnerType {
  MOCK,
  LLAMA,
  MOCK_PIPELINE,
  PIPELINE_MANAGER,
  PREFILL,
  TT_SDXL_GENERATE,
  TT_SDXL_IMAGE_TO_IMAGE,
  TT_SDXL_EDIT,
};

enum class Model {
  DEEPSEEK_R1_0528,
  LLAMA_3_1_8B_INSTRUCT,
  KIMI_K2_6,
};

struct ModelMapping {
  Model model;
  std::string_view name;
};

static constexpr ModelMapping MODEL_MAPPINGS[] = {
    {Model::DEEPSEEK_R1_0528, "deepseek-ai/DeepSeek-R1-0528"},
    {Model::LLAMA_3_1_8B_INSTRUCT, "meta-llama/Llama-3.1-8B-Instruct"},
    {Model::KIMI_K2_6, "moonshotai/Kimi-K2.6"},
};

inline std::string toString(Model m) {
  for (const auto& entry : MODEL_MAPPINGS) {
    if (entry.model == m) return std::string(entry.name);
  }
  throw std::invalid_argument("Cannot match model to string");
}

inline std::string toString(ModelRunnerType m) {
  switch (m) {
    case ModelRunnerType::MOCK:
      return "mock";
    case ModelRunnerType::LLAMA:
      return "llama";
    case ModelRunnerType::MOCK_PIPELINE:
      return "mock_pipeline";
    case ModelRunnerType::PIPELINE_MANAGER:
      return "pipeline_manager";
    case ModelRunnerType::PREFILL:
      return "prefill";
    case ModelRunnerType::TT_SDXL_GENERATE:
      return "tt_sdxl_generate";
    case ModelRunnerType::TT_SDXL_IMAGE_TO_IMAGE:
      return "tt_sdxl_image_to_image";
    case ModelRunnerType::TT_SDXL_EDIT:
      return "tt_sdxl_edit";
  }
  return "unknown";
}

// Matches the `ModelRunners` enum values in tt-media-server/config/constants.py
inline std::string toClientRunnerName(ModelRunnerType m) {
  switch (m) {
    case ModelRunnerType::TT_SDXL_GENERATE:
      return "tt-sdxl-trace";
    case ModelRunnerType::TT_SDXL_IMAGE_TO_IMAGE:
      return "tt-sdxl-image-to-image";
    case ModelRunnerType::TT_SDXL_EDIT:
      return "tt-sdxl-edit";
    case ModelRunnerType::MOCK:
    case ModelRunnerType::LLAMA:
    case ModelRunnerType::MOCK_PIPELINE:
    case ModelRunnerType::PIPELINE_MANAGER:
    case ModelRunnerType::PREFILL:
      return "";
  }
  return "";
}

inline Model modelFromString(const std::string_view& v) {
  for (const auto& entry : MODEL_MAPPINGS) {
    if (entry.name == v) return entry.model;
  }
  throw std::invalid_argument("Invalid model: " + std::string(v));
}

enum class ResponseFormatType : uint8_t {
  TEXT = 0,
  JSON_OBJECT = 1,
  JSON_SCHEMA = 2
};

enum class SchedulingPolicy {
  PREFILL_FIRST,
  MAX_OCCUPANCY,
};

/** Parse SCHEDULING_POLICY; empty or unknown -> PREFILL_FIRST. Expects
 * lowercase input. */
inline SchedulingPolicy schedulingPolicyFromString(const std::string& v) {
  if (v == "max_occupancy") return SchedulingPolicy::MAX_OCCUPANCY;
  return SchedulingPolicy::PREFILL_FIRST;
}

}  // namespace tt::config
