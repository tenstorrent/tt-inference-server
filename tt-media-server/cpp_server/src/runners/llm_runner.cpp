#include "runners/llm_runner.hpp"
#include "runners/llm_runner/debug.hpp"

#include <cassert>
#include <iostream>

namespace tt::runners {
  using namespace llm_engine;

LLMRunner::LLMRunner(const Config& config, ipc::TokenRingBuffer<65536>* result_queue, ITaskQueue* task_queue)
    : config_(config), result_queue_(result_queue) {
  LLM_ENGINE_LOG("llm_engine") << "construct" << std::endl;

  scheduler_ = std::make_unique<Scheduler>(config_, task_queue);

  auto decode_cb = [this](const TokenResult& result) {
    decode_queue_.push(result);
  };

  model_runner_ = make_model_runner(config_, std::move(decode_cb));
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
      auto shared = ipc::SharedToken{
          .token_index = 0,
          .flags = static_cast<uint32_t>(ipc::SharedToken::FLAG_FINAL |
                                         ipc::SharedToken::FLAG_ERROR),
          .token_id = 0,
          .task_id = {},
          .padding = {},
      };
      strncpy(shared.task_id, dr.task_id.id.c_str(), sizeof(shared.task_id) - 1);
      shared.task_id[sizeof(shared.task_id) - 1] = '\0';
      result_queue_->push(shared);
      continue;
    }

    std::vector<Sequence*> seqs = {seq};
    std::vector<int64_t> token_ids = {static_cast<int64_t>(dr.token_id)};
    scheduler_->postprocess(seqs, token_ids);

    bool finished = seq->is_finished();

    auto shared = ipc::SharedToken{
        .token_index = 0,
        .flags = static_cast<uint32_t>(finished ? ipc::SharedToken::FLAG_FINAL : 0),
        .token_id = dr.token_id,
        .task_id = {},
        .padding = {},
    };
    strncpy(shared.task_id, dr.task_id.id.c_str(), sizeof(shared.task_id) - 1);
    shared.task_id[sizeof(shared.task_id) - 1] = '\0';
    result_queue_->push(shared);

    if (finished) {
      LLM_ENGINE_LOG("llm_engine") << "finished task_id=" << seq->task_id
                                   << " completion_tokens=" << seq->num_completion_tokens() << std::endl;
      scheduler_->removeSequence(dr.task_id);
    }
  }
}

}  // namespace tt::runners
