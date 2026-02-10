#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/debug.hpp"

#include <chrono>

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
    : config_(config),
      dummy_token_((config.eos == 0) ? 1 : 0),
      decode_callback_(std::move(callback)),
      reader_thread_([this] { reader_loop(); }) {}

ModelRunnerStub::~ModelRunnerStub() {
  exit();
}

void ModelRunnerStub::reader_loop() {
  int channel = 0;
  int dummy_token = 100;
  while (!stop_.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    decode_callback_({channel, ++dummy_token});
    channel = (channel + 1) % NUM_DECODE_CHANNELS;
  }
}

void ModelRunnerStub::run(const std::vector<Sequence*>& seqs,
                          bool is_prefill) {
  LLM_ENGINE_LOG("model_runner") << (is_prefill ? "prefill" : "decode")
                               << " batch_size=" << seqs.size() << std::endl;

  if (is_prefill) {
    for (Sequence* seq : seqs) {
      decode_callback_({seq->seq_id, ++dummy_token_});
    }
  }
}

void ModelRunnerStub::exit() {
  if (stop_.exchange(true)) return;
  if (reader_thread_.joinable()) reader_thread_.join();
  LLM_ENGINE_LOG("model_runner") << "exit" << std::endl;
}

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback) {
  return std::make_unique<ModelRunnerStub>(config, std::move(callback));
}

}  // namespace llm_engine
