#include "runners/llm_runner.hpp"
#include "runners/llm_runner/debug.hpp"

#include <cassert>

namespace tt::runners {
  using namespace llm_engine;

LLMRunner::LLMRunner(const Config& config, TokenCallback on_token, ITaskQueue* task_queue,
                     ModelRunnerFactory factory)
    : config_(config), on_token_(std::move(on_token)) {
  LLM_ENGINE_LOG("llm_engine") << "construct" << std::endl;

  scheduler_ = std::make_unique<Scheduler>(config_, task_queue);

  auto decode_cb = [this](const DecodeResult& result) {
    decode_queue_.push(result);
  };
  if (factory) {
    model_runner_ = factory(config_, std::move(decode_cb));
  } else {
    model_runner_ = make_model_runner(config_, std::move(decode_cb));
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

    if (dr.is_error) {
      LLM_ENGINE_LOG("llm_engine") << "error task_id=" << seq->task_id << std::endl;
      scheduler_->removeSequence(dr.task_id);
      on_token_(dr.task_id, 0, /*finished=*/true, /*is_stop=*/false, /*is_error=*/true);
      continue;
    }

    std::vector<Sequence*> seqs = {seq};
    std::vector<int64_t> token_ids = {dr.token_id};
    scheduler_->postprocess(seqs, token_ids);

    bool finished = seq->is_finished();
    bool is_stop = finished && scheduler_->is_stop_token(dr.token_id);
    on_token_(dr.task_id, dr.token_id, finished, is_stop, /*is_error=*/false);

    if (finished) {
      LLM_ENGINE_LOG("llm_engine") << "finished task_id=" << seq->task_id
                                   << " completion_tokens=" << seq->num_completion_tokens() << std::endl;
      scheduler_->removeSequence(dr.task_id);
    }
  }
}

}  // namespace tt::runners
