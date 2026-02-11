#include "llm_engine/engine/llm_engine.hpp"
#include "llm_engine/engine/debug.hpp"
#include "llm_engine/engine/tracy.hpp"
#include <cassert>

namespace llm_engine {

LLMEngine::LLMEngine(const Config& config, TokenCallback on_token)
    : config_(config), on_token_(std::move(on_token)) {
  LLM_ENGINE_LOG("llm_engine") << "construct" << std::endl;
  auto decode_cb = [this](const DecodeResult& result) {
    decode_queue_.push(result);
  };
  model_runner_ = make_model_runner(config_, std::move(decode_cb));
  scheduler_ = std::make_unique<Scheduler>(config_);
  if (config_.eos < 0) {
    config_.eos = 0;
  }
}

LLMEngine::~LLMEngine() {
  exit();
}

void LLMEngine::exit() {
  if (model_runner_) {
    LLM_ENGINE_LOG("llm_engine") << "exit" << std::endl;
    model_runner_->exit();
  }
}

void LLMEngine::run() {
  LLM_ENGINE_LOG("llm_engine") << "run" << std::endl;
  while (!stopped_.load(std::memory_order_relaxed)) {
    ZoneTransientN(zone_run_loop, "LLMEngine run loop", true);
    FrameMark;
    step();
  }
  LLM_ENGINE_LOG("llm_engine") << "run done" << std::endl;
}

void LLMEngine::stop() {
  stopped_.store(true, std::memory_order_relaxed);
}

void LLMEngine::step() {
  ZoneTransientN(zone_step, "step", true);
  drain_decode_results();

  std::vector<Sequence*> seqs;
  bool is_prefill = false;
  {
    ZoneTransientN(zone_schedule, "schedule", true);
    auto result = scheduler_->schedule();
    seqs = result.first;
    is_prefill = result.second;
  }
  if (seqs.empty()) return;

  model_runner_->run(seqs, is_prefill);

  LLM_ENGINE_LOG("llm_engine") << "step " << (is_prefill ? "prefill" : "decode")
                               << " n=" << seqs.size() << std::endl;
}

void LLMEngine::drain_decode_results() {
  ZoneTransientN(zone_drain, "drain_decode_results", true);
  for (const auto& dr : decode_queue_.drain()) {
    ZoneTransientN(zone_drain_one, "drain one result", true);
    ZoneTextVF(zone_drain_one, "seq_id=%d token_id=%lld", dr.seq_id, static_cast<long long>(dr.token_id));
    Sequence* seq = scheduler_->find_sequence(dr.seq_id);
    assert(seq);
    assert(seq->status_ == SequenceStatus::IN_FLIGHT);

    std::vector<Sequence*> seqs = {seq};
    std::vector<int64_t> token_ids = {dr.token_id};
    scheduler_->postprocess(seqs, token_ids);

    bool finished = seq->is_finished();
    on_token_(dr.seq_id, dr.token_id, finished);

    if (finished) {
      LLM_ENGINE_LOG("llm_engine") << "finished seq_id=" << seq->seq_id
                                   << " completion_tokens=" << seq->num_completion_tokens() << std::endl;
    }
  }
}

}  // namespace llm_engine
