#include "runners/llm_runner/model_runner.hpp"

#include <stdexcept>

#ifdef USE_METAL_CPP_LIB
#include "runners/llama_model_runner.hpp"
#endif

#include <iostream>

namespace llm_engine {

std::unique_ptr<IModelRunner> make_mock_model_runner(const Config& config,
                                                     DecodeCallback callback);
#ifdef USE_METAL_CPP_LIB
std::unique_ptr<IModelRunner> make_llama_model_runner(const Config& config,
                                                      DecodeCallback callback);
#endif

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback) {
  switch (config.runner_type) {
    case ModelRunnerType::Mock:
      return make_mock_model_runner(config, std::move(callback));
#ifdef USE_METAL_CPP_LIB
    case ModelRunnerType::Llama:
      return make_llama_model_runner(config, std::move(callback));
#endif
    default:
      throw std::invalid_argument("Invalid model runner type");
  }
}

}  // namespace llm_engine
