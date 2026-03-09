#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "profiling/tracy.hpp"

namespace llm_engine {

using DecodeCallback = std::function<void(const TokenResult&)>;

class DecodeQueue {
 public:
  void push(const TokenResult& result);
  std::vector<TokenResult> drain();

 private:
  TracyLockable(std::mutex, mutex_);
  std::vector<TokenResult> pending_;
};

class IModelRunner {
 public:
  virtual ~IModelRunner() = default;
  virtual void run(const std::vector<Sequence*>& seqs, bool is_prefill) = 0;
  virtual void exit() = 0;
};

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback);

}  // namespace llm_engine
