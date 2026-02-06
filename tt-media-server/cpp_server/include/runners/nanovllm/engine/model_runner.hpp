#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "nanovllm/config.hpp"
#include "nanovllm/engine/sequence.hpp"

namespace nanovllm {

class IModelRunner {
 public:
  virtual ~IModelRunner() = default;
  virtual std::vector<int64_t> run(const std::vector<Sequence*>& seqs,
                                   bool is_prefill) = 0;
  virtual void exit() = 0;
};

class ModelRunnerStub : public IModelRunner {
 public:
  explicit ModelRunnerStub(const Config& config);
  std::vector<int64_t> run(const std::vector<Sequence*>& seqs,
                           bool is_prefill) override;
  void exit() override;

 private:
  Config config_;
  int eos_;
};

std::unique_ptr<IModelRunner> make_model_runner(const Config& config);

}  // namespace nanovllm
