#include "runners/llm_runner/model_runner.hpp"

#include <stdexcept>

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

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback) {
  switch (config.runner_type) {
    case ModelRunnerType::Mock:
      return make_mock_model_runner(config, std::move(callback));
    default:
      throw std::invalid_argument("Invalid model runner type");
  }
}

}  // namespace llm_engine
