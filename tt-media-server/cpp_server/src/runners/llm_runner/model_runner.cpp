#include "runners/llm_runner/model_runner.hpp"

#include <stdexcept>

#ifdef USE_METAL_CPP_LIB
#include "runners/llama_model_runner.hpp"
#endif

#include <iostream>

namespace tt::runners::llm_engine {

using Config = tt::config::LLMConfig;
using ModelRunnerType = tt::config::ModelRunnerType;

std::unique_ptr<IModelRunner> makeMockModelRunner(const Config& config,
                                                  DecodeCallback callback);
#ifdef USE_METAL_CPP_LIB
std::unique_ptr<IModelRunner> makeLlamaModelRunner(const Config& config,
                                                   DecodeCallback callback);
#endif

std::unique_ptr<IModelRunner> makeModelRunner(const Config& config,
                                              DecodeCallback callback) {
  switch (config.runner_type) {
    case ModelRunnerType::MOCK:
      return makeMockModelRunner(config, std::move(callback));
#ifdef USE_METAL_CPP_LIB
    case ModelRunnerType::LLAMA:
      return makeLlamaModelRunner(config, std::move(callback));
#endif
    default:
      throw std::invalid_argument("Invalid model runner type");
  }
}

}  // namespace tt::runners::llm_engine
