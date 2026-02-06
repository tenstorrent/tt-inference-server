#include "nanovllm/engine/model_runner.hpp"
#include "nanovllm/engine/debug.hpp"

namespace nanovllm {

ModelRunnerStub::ModelRunnerStub(const Config& config)
    : config_(config), eos_(config.eos) {}

std::vector<int64_t> ModelRunnerStub::run(const std::vector<Sequence*>& seqs,
                                          bool is_prefill) {
  NANOVLLM_LOG("model_runner") << (is_prefill ? "prefill" : "decode")
                               << " batch_size=" << seqs.size() << std::endl;
  const int64_t dummy = (eos_ == 0) ? 1 : 0;
  std::vector<int64_t> token_ids(seqs.size(), dummy);
  return token_ids;
}

void ModelRunnerStub::exit() {
  NANOVLLM_LOG("model_runner") << "exit" << std::endl;
}

std::unique_ptr<IModelRunner> make_model_runner(const Config& config) {
  return std::make_unique<ModelRunnerStub>(config);
}

}  // namespace nanovllm
