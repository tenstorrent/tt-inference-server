#include "runners/llm_runner/model_runner.hpp"

#include <stdexcept>

#ifdef USE_METAL_CPP_LIB
#include "runners/llm_runner/model_runners/llama_model_runner.hpp"
#endif

#include <iostream>

namespace llm_engine {

void DecodeQueue::push(const TokenResult& result) {
  std::lock_guard lock(mutex_);
  pending_.push_back(result);
}

std::vector<TokenResult> DecodeQueue::drain() {
  std::lock_guard lock(mutex_);
  std::vector<TokenResult> out;
  out.swap(pending_);
  return out;
}

std::unique_ptr<IModelRunner> make_mock_model_runner(const Config& config,
                                                     DecodeCallback callback);
std::unique_ptr<IModelRunner> make_ttrun_model_runner(const Config& config,
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
    case ModelRunnerType::TtRun:
      return make_ttrun_model_runner(config, std::move(callback));
#ifdef USE_METAL_CPP_LIB
    case ModelRunnerType::Llama:
      return make_llama_model_runner(config, std::move(callback));
#endif
    default:
      throw std::invalid_argument("Invalid model runner type");
  }
}

}  // namespace llm_engine
