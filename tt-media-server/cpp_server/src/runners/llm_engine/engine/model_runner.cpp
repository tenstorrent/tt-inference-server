#include "llm_engine/engine/model_runner.hpp"
#include "llm_engine/engine/debug.hpp"
#include "llm_engine/engine/tracy.hpp"
#include <thread>

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
  TracySetThreadName("receiver");
  while (!stop_.load(std::memory_order_relaxed)) {
    ZoneTransientN(zone_iter, "reader_loop iteration", true);
    std::vector<DecodeResult> work;
    {
      ZoneTransientN(zone_drain_q, "reader_loop drain work_queue", true);
      std::lock_guard<std::mutex> lock(work_mutex_);
      work.swap(work_queue_);
    }
    for (const auto& item : work) {
      ZoneTransientN(zone_cb, "reader_loop decode_callback", true);
      ZoneTextVF(zone_cb, "seq_id=%d token_id=%lld", item.seq_id, static_cast<long long>(item.token_id));
      decode_callback_({item.seq_id, item.token_id});
    }
    if (work.empty()) {
      std::this_thread::yield();
    }
  }
}

void ModelRunnerStub::run(const std::vector<Sequence*>& seqs,
                          bool is_prefill) {
  ZoneTransientN(zone_run, "model_runner run", true);
  LLM_ENGINE_LOG("model_runner") << (is_prefill ? "prefill" : "decode")
                               << " batch_size=" << seqs.size() << std::endl;

  if (is_prefill) {
    ZoneTransientN(zone_prefill, "run prefill", true);
    for (Sequence* seq : seqs) {
      ZoneTransientN(zone_prefill_seq, "prefill seq", true);
      ZoneTextVF(zone_prefill_seq, "seq_id=%d token_id=%lld", seq->seq_id, static_cast<long long>(seq->last_token + 1));
      decode_callback_({seq->seq_id, seq->last_token + 1});
    }
  } else {
    ZoneTransientN(zone_decode, "run decode (push to work_queue)", true);
    std::lock_guard<std::mutex> lock(work_mutex_);
    for (Sequence* seq : seqs) {
      ZoneTransientN(zone_decode_seq, "decode push seq", true);
      ZoneTextVF(zone_decode_seq, "seq_id=%d token_id=%lld", seq->seq_id, static_cast<long long>(seq->last_token + 1));
      work_queue_.push_back({seq->seq_id, seq->last_token + 1});
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
