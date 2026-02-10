#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/debug.hpp"

#include <thread>

namespace llm_engine {

ModelRunnerStub::ModelRunnerStub(const Config& config, DecodeCallback callback)
    : config_(config),
      eos_(config.eos),
      decode_callback_(std::move(callback)) {}

std::vector<int64_t> ModelRunnerStub::run(const std::vector<Sequence*>& seqs,
                                          bool is_prefill) {
  LLM_ENGINE_LOG("model_runner") << (is_prefill ? "prefill" : "decode")
                               << " batch_size=" << seqs.size() << std::endl;

  const int64_t dummy = (eos_ == 0) ? 1 : 0;

  if (is_prefill) {
    return std::vector<int64_t>(seqs.size(), dummy);
  }

  // Blitz decode: send (token_id, position_id, seq_id) to device,
  // then tokens arrive asynchronously via the device-to-host reader thread.
  // Simulate by firing the callback from a short-lived reader thread.
  std::vector<DecodeResult> results;
  results.reserve(seqs.size());
  for (const auto* seq : seqs) {
    results.push_back({seq->seq_id, dummy});
  }

  std::thread reader{[results = std::move(results),
                      cb = this->decode_callback_]() {
    for (const auto& r : results) {
      cb(r);
    }
  }};
  // In production the reader thread is long-running; here we join for simulation.
  reader.join();

  return {};
}

void ModelRunnerStub::exit() {
  LLM_ENGINE_LOG("model_runner") << "exit" << std::endl;
}

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback) {
  return std::make_unique<ModelRunnerStub>(config, std::move(callback));
}

}  // namespace llm_engine
