#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/debug.hpp"
#include "llm_engine/engine/device_backend.hpp"

#include <cstring>

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

ModelRunnerStub::ModelRunnerStub(const Config& config, DecodeCallback callback,
                                std::unique_ptr<IDeviceBackend> backend)
    : config_(config),
      decode_callback_(std::move(callback)),
      backend_(std::move(backend)) {
  backend_->init();
  reader_thread_ = std::thread([this] { reader_loop(); });
}

ModelRunnerStub::~ModelRunnerStub() {
  exit();
}

void ModelRunnerStub::reader_loop() {
  std::vector<char> output(Sequence::page_size(), 0);
  while (!stop_.load(std::memory_order_relaxed)) {
    if (!backend_->read(output.data(), 1)) break;
    if (stop_.load(std::memory_order_relaxed)) break;
    SequenceID seq_id = SequenceID::deserialize(output.data(), SequenceID::kSerializedSize);
    int64_t last_token;
    std::memcpy(&last_token, output.data() + SequenceID::kSerializedSize, sizeof(last_token));
    decode_callback_(DecodeResult{seq_id, last_token});
  }
}

void ModelRunnerStub::run(const std::vector<Sequence*>& seqs,
                          bool is_prefill) {
  LLM_ENGINE_LOG("model_runner") << (is_prefill ? "prefill" : "decode")
                               << " batch_size=" << seqs.size() << std::endl;

  if (is_prefill) {
    for (Sequence* seq : seqs) {
      decode_callback_({seq->seq_id, seq->last_token + 1});
    }
  } else {
    backend_->write(seqs[0]->to_h2d_input().data(), 1);
  }
}

void ModelRunnerStub::exit() {
  if (stop_.exchange(true)) return;
  backend_->terminate();
  if (reader_thread_.joinable()) reader_thread_.join();
  LLM_ENGINE_LOG("model_runner") << "exit" << std::endl;
}

std::unique_ptr<IModelRunner> make_model_runner(const Config& config,
                                                DecodeCallback callback) {
  auto backend = make_device_backend(config, config.use_real_device);
  return std::make_unique<ModelRunnerStub>(config, std::move(callback), std::move(backend));
}

}  // namespace llm_engine
