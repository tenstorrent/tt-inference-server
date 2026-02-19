#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/debug.hpp"
#include "llm_engine/engine/device_backend.hpp"
#include "llm_engine/engine/sequence.hpp"

namespace llm_engine {

void DecodeQueue::push(const DecodeResult& result) {
  std::lock_guard lock(mutex_);
  pending_.push_back(result);
}

std::vector<DecodeResult> DecodeQueue::drain() {
  std::lock_guard lock(mutex_);
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
  DecodeResult result;
  while (!stop_.load(std::memory_order_relaxed)) {
    if (!backend_->read(&result)) break;
    if (stop_.load(std::memory_order_relaxed)) break;
    decode_callback_(result);
  }
}

void ModelRunnerStub::run(const std::vector<Sequence*>& seqs,
                          bool is_prefill) {
  LLM_ENGINE_LOG("model_runner") << (is_prefill ? "prefill" : "decode")
                               << " batch_size=" << seqs.size() << std::endl;

  if (is_prefill) {
    for (Sequence* seq : seqs) {
      decode_callback_({seq->task_id, seq->last_token + 1});
    }
  } else {
    backend_->write(*seqs[0]);
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
  auto backend = make_device_backend(config);
  return std::make_unique<ModelRunnerStub>(config, std::move(callback), std::move(backend));
}

}  // namespace llm_engine
