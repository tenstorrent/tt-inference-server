#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/sequence.hpp"

namespace tt::runners::llm_engine {

using DecodeCallback = std::function<void(const tt::domain::TokenResult&)>;

class IModelRunner {
 public:
  virtual ~IModelRunner() = default;
  virtual void run(const std::vector<tt::domain::Sequence*>& seqs,
                   bool isPrefill) = 0;
  virtual void exit() = 0;
};

std::unique_ptr<IModelRunner> makeModelRunner(
    const tt::config::LLMConfig& config, DecodeCallback callback);

}  // namespace tt::runners::llm_engine
