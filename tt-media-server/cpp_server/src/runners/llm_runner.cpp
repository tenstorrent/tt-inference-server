#include "runners/llm_runner.hpp"
#include "llm_engine/engine/debug.hpp"

#include <cassert>

namespace llm_engine {

LLMRunner::LLMRunner(const Config& config, TokenCallback on_token, std::unique_ptr<Scheduler> scheduler)
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

LLMRunner::~LLMRunner() {
  exit();
}

void LLMRunner::exit() {
  if (model_runner_) {
    LLM_ENGINE_LOG("llm_engine") << "exit" << std::endl;
    model_runner_->exit();
  }
}

void LLMRunner::run() {
  LLM_ENGINE_LOG("llm_engine") << "run" << std::endl;
  while (!stopped_.load(std::memory_order_relaxed)) {
    step();
  }
  LLM_ENGINE_LOG("llm_engine") << "run done" << std::endl;
}

void LLMRunner::stop() {
  stopped_.store(true, std::memory_order_relaxed);
}



void LLMRunner::step() {
  drain_decode_results();

  auto [seqs, is_prefill] = scheduler_->schedule();
  if (seqs.empty()) return;

  model_runner_->run(seqs, is_prefill);

  LLM_ENGINE_LOG("llm_engine") << "step " << (is_prefill ? "prefill" : "decode")
                               << " n=" << seqs.size() << std::endl;
}

void LLMRunner::drain_decode_results() {
  for (const auto& dr : decode_queue_.drain()) {
    Sequence* seq = scheduler_->find_sequence(dr.task_id);
    assert(seq);
    assert(seq->status_ == SequenceStatus::IN_FLIGHT);

    std::vector<Sequence*> seqs = {seq};
    std::vector<int64_t> token_ids = {dr.token_id};
    scheduler_->postprocess(seqs, token_ids);

    bool finished = seq->is_finished();
    on_token_(dr.task_id, dr.token_id, finished);

    if (finished) {
      LLM_ENGINE_LOG("llm_engine") << "finished task_id=" << seq->task_id
                                   << " completion_tokens=" << seq->num_completion_tokens() << std::endl;
      scheduler_->removeSequence(dr.task_id);
    }
  }
}

}  // namespace llm_engine
