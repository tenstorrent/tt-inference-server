#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/spoofed_blitz_decode.hpp"
#include "llm_engine/engine/debug.hpp"

namespace llm_engine {

void DecodeQueue::push(const DecodeResult& result) {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.push_back(result);
}

std::vector<DecodeResult> DecodeQueue::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<DecodeResult> out;
  out.swap(pending_);
  return out;
}

ModelRunnerStub::ModelRunnerStub(const Config& config, DecodeCallback callback)
    : config_{config},
      decode_callback_{std::move(callback)},
      spoofed_decode_{std::make_unique<SpoofedBlitzDecode>(config, decode_callback_)} {
  spoofed_decode_->run();
}

ModelRunnerStub::~ModelRunnerStub() {
  exit();
}

void ModelRunnerStub::run(const std::vector<Sequence*>& seqs, bool is_prefill) {
  LLM_ENGINE_LOG("model_runner") << (is_prefill ? "prefill" : "decode")
                                 << " batch_size=" << seqs.size() << std::endl;

  if (is_prefill) {
    for (Sequence* seq : seqs) {
      decode_callback_({seq->seq_id, seq->last_token + 1});
    }
  } else {
    spoofed_decode_->decode(seqs);
  }
}

void ModelRunnerStub::exit() {
  spoofed_decode_->exit();
  LLM_ENGINE_LOG("model_runner") << "exit" << std::endl;
}

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback) {
  return std::make_unique<ModelRunnerStub>(config, std::move(callback));
}

}  // namespace llm_engine
