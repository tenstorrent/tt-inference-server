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
  while (!stop_.load(std::memory_order_relaxed)) {
    std::vector<DecodeResult> work;
    {
      std::lock_guard<std::mutex> lock(work_mutex_);
      work.swap(work_queue_);
    }
    for (const auto& item : work) {
      decode_callback_({item.task_id, item.token_id});
    }
    if (work.empty()) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
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
    // h2d
    std::lock_guard<std::mutex> lock(work_mutex_);
    for (Sequence* seq : seqs) {
      work_queue_.push_back({seq->task_id, seq->last_token + 1});
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
