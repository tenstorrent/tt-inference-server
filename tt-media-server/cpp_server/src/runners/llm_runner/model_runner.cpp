#include "runners/llm_runner/model_runner.hpp"

#include <stdexcept>

#ifdef USE_METAL_CPP_LIB
#include "runners/llama_model_runner.hpp"
#endif
#include <iostream>

#include "runners/qwen_model_runner.hpp"

namespace tt::runners::llm_engine {

using Config = tt::config::LLMConfig;
using ModelRunnerType = tt::config::ModelRunnerType;

std::unique_ptr<IModelRunner> makeMockModelRunner(const Config& config,
                                                  DecodeCallback callback);
#ifdef USE_METAL_CPP_LIB
std::unique_ptr<IModelRunner> makeLlamaModelRunner(const Config& config,
                                                   DecodeCallback callback);
#endif
std::unique_ptr<IModelRunner> makeQwenModelRunner(const Config& config,
                                                  DecodeCallback callback);

std::unique_ptr<IModelRunner> makeModelRunner(const Config& config,
                                              DecodeCallback callback) {
  switch (config.runner_type) {
    case ModelRunnerType::MOCK:
#ifndef ENABLE_BLAZE
    case ModelRunnerType::MOCK_PIPELINE:
    case ModelRunnerType::PIPELINE_MANAGER:
#endif
      return makeMockModelRunner(config, std::move(callback));
#ifdef USE_METAL_CPP_LIB
    case ModelRunnerType::LLAMA:
      return makeLlamaModelRunner(config, std::move(callback));
#endif
    case ModelRunnerType::QWEN:
      return makeQwenModelRunner(config, std::move(callback));
    default:
      throw std::invalid_argument("Invalid model runner type: " +
                                  toString(config.runner_type));
  }
}

}  // namespace tt::runners::llm_engine
