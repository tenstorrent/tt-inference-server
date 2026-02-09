#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "llm_engine/config.hpp"
#include "llm_engine/engine/sequence.hpp"

namespace llm_engine {

/** Called when token IDs for the batch are ready (sync for prefill, async for decode). */
using OnTokensCallback =
    std::function<void(const std::vector<Sequence*>& seqs, std::vector<int64_t> token_ids)>;

class IModelRunner {
 public:
  virtual ~IModelRunner() = default;
  /** Submits batch; on_tokens(seqs, token_ids) is invoked when results are ready. */
  virtual void run(const std::vector<Sequence*>& seqs,
                   bool is_prefill,
                   OnTokensCallback on_tokens) = 0;
  virtual void exit() = 0;
};

class ModelRunnerStub : public IModelRunner {
 public:
  explicit ModelRunnerStub(const Config& config);
  void run(const std::vector<Sequence*>& seqs,
           bool is_prefill,
           OnTokensCallback on_tokens) override;
  void exit() override;

 private:
  Config config_;
  int eos_;
};

std::unique_ptr<IModelRunner> make_model_runner(const Config& config);

}  // namespace llm_engine
