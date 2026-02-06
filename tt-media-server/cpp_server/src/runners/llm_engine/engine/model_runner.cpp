#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/debug.hpp"

namespace llm_engine {

ModelRunnerStub::ModelRunnerStub(const Config& config)
    : config_(config), eos_(config.eos) {}

std::vector<int64_t> ModelRunnerStub::run(const std::vector<Sequence*>& seqs,
                                          bool is_prefill) {
  LLM_ENGINE_LOG("model_runner") << (is_prefill ? "prefill" : "decode")
                               << " batch_size=" << seqs.size() << std::endl;
  const int64_t dummy = (eos_ == 0) ? 1 : 0;
  std::vector<int64_t> token_ids(seqs.size(), dummy);
  return token_ids;
}

void ModelRunnerStub::exit() {
  LLM_ENGINE_LOG("model_runner") << "exit" << std::endl;
}

std::unique_ptr<IModelRunner> make_model_runner(const Config& config) {
  return std::make_unique<ModelRunnerStub>(config);
}

}  // namespace llm_engine
