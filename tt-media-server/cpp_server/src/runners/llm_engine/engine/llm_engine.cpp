#include "llm_engine/engine/llm_engine.hpp"
#include "llm_engine/engine/debug.hpp"

#include <cassert>
#include <chrono>
#include <thread>

namespace llm_engine {

LLMEngine::LLMEngine(const Config& config, TokenCallback on_token, std::unique_ptr<Scheduler> scheduler)
    : config_(config), on_token_(std::move(on_token)), scheduler_(std::move(scheduler)) {
  LLM_ENGINE_LOG("llm_engine") << "construct" << std::endl;
  auto decode_cb = [this](const DecodeResult& result) {
    decode_queue_.push(result);
  };
  model_runner_ = make_model_runner(config_, std::move(decode_cb));
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
    step();
  }
  LLM_ENGINE_LOG("llm_engine") << "run done" << std::endl;
}

void LLMEngine::stop() {
  stopped_.store(true, std::memory_order_relaxed);
}



void LLMEngine::step() {
  drain_decode_results();

  auto [seqs, is_prefill] = scheduler_->schedule();
  if (seqs.empty()) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    return;
  }

  model_runner_->run(seqs, is_prefill);

  LLM_ENGINE_LOG("llm_engine") << "step " << (is_prefill ? "prefill" : "decode")
                               << " n=" << seqs.size() << std::endl;
}

void LLMEngine::drain_decode_results() {
  for (const auto& dr : decode_queue_.drain()) {
    Sequence* seq = scheduler_->find_sequence(dr.seq_id);
    
    std::string status;
    switch (seq->status_) {
      case SequenceStatus::WAITING:
        status = "WAITING";
        break;
      case SequenceStatus::RUNNING:
        status = "RUNNING";
        break;
      case SequenceStatus::IN_FLIGHT:
        status = "IN_FLIGHT";
        break;
      case SequenceStatus::FINISHED:
        status = "FINISHED";
        break;
    }
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
