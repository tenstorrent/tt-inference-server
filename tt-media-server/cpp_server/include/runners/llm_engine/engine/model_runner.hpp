#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "llm_engine/config.hpp"
#include "llm_engine/engine/sequence.hpp"

namespace llm_engine {

struct DecodeResult {
  int seq_id;
  int64_t token_id;
};

// Invoked from the device-to-host reader thread when a token is generated.
using DecodeCallback = std::function<void(const DecodeResult&)>;

class IModelRunner {
 public:
  virtual ~IModelRunner() = default;
  virtual std::vector<int64_t> run(const std::vector<Sequence*>& seqs,
                                   bool is_prefill) = 0;
  virtual void exit() = 0;
};

class ModelRunnerStub : public IModelRunner {
 public:
  ModelRunnerStub(const Config& config, DecodeCallback callback);
  std::vector<int64_t> run(const std::vector<Sequence*>& seqs,
                           bool is_prefill) override;
  void exit() override;

 private:
  Config config_;
  int eos_;
  DecodeCallback decode_callback_;
};

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback);

}  // namespace llm_engine
