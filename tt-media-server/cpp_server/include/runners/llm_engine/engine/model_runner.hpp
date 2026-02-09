#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "llm_engine/config.hpp"
#include "llm_engine/engine/sequence.hpp"

namespace llm_engine {

/** One (token_id, user_id) from the pipeline; position_id is not used on receive. */
using TokenEntry = std::pair<int64_t, int>;
/** Called with one entry per response (sync for prefill batch, async for decode). Loopback returns what we pushed. */
using OnTokensCallback = std::function<void(std::vector<TokenEntry>)>;

class IModelRunner {
 public:
  virtual ~IModelRunner() = default;
  /** Set once; invoked when token IDs for a batch are ready (sync for prefill, async for decode). */
  virtual void set_on_tokens_callback(OnTokensCallback on_tokens) = 0;
  /** Submits batch; runner invokes the callback set via set_on_tokens_callback when results are ready. */
  virtual void run(const std::vector<Sequence*>& seqs, bool is_prefill) = 0;
  virtual void exit() = 0;
};

class ModelRunnerStub : public IModelRunner {
 public:
  explicit ModelRunnerStub(const Config& config);
  void set_on_tokens_callback(OnTokensCallback on_tokens) override;
  void run(const std::vector<Sequence*>& seqs, bool is_prefill) override;
  void exit() override;

 private:
  Config config_;
  int eos_;
  OnTokensCallback on_tokens_callback_;
};

std::unique_ptr<IModelRunner> make_model_runner(const Config& config);

}  // namespace llm_engine
