// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/deepseek_runner.hpp"
#include "runners/llm_runner/debug.hpp"

#include <cstring>
#include <thread>

namespace tt::runners {

using namespace llm_engine;

DeepSeekRunner::DeepSeekRunner(const Config& config,
                               ipc::TokenRingBuffer<65536>* result_queue,
                               ITaskQueue* task_queue)
    : config_(config), result_queue_(result_queue), task_queue_(task_queue) {
  LLM_ENGINE_LOG("deepseek") << "construct" << std::endl;

  auto decode_cb = [this](const TokenResult& result) {
    std::lock_guard lock(in_flight_mutex_);
    auto it = in_flight_.find(result.task_id);
    if (it == in_flight_.end()) return;

    auto& seq = it->second;
    seq.tokens_received++;

    bool finished = (seq.tokens_received >= seq.max_tokens) ||
                    (!seq.ignore_eos && result.token_id == static_cast<uint64_t>(config_.eos));

    auto shared = ipc::SharedToken{
        .token_index = 0,
        .flags = static_cast<uint32_t>(finished ? ipc::SharedToken::FLAG_FINAL : 0),
        .token_id = result.token_id,
        .task_id = {},
        .padding = {},
    };
    strncpy(shared.task_id, seq.task_id_str.c_str(), sizeof(shared.task_id) - 1);
    shared.task_id[sizeof(shared.task_id) - 1] = '\0';

    while (!result_queue_->push(shared)) {
      std::this_thread::yield();
    }

    if (finished) {
      LLM_ENGINE_LOG("deepseek") << "finished task_id=" << seq.task_id_str
                                 << " tokens=" << seq.tokens_received << std::endl;
      in_flight_.erase(it);
      in_flight_count_.fetch_sub(1, std::memory_order_relaxed);
    }
  };

  model_runner_ = make_deepseek_model_runner(config_, std::move(decode_cb));
}

DeepSeekRunner::~DeepSeekRunner() {
  if (model_runner_) {
    model_runner_->exit();
  }
}

void DeepSeekRunner::run() {
  LLM_ENGINE_LOG("deepseek") << "run" << std::endl;
  while (!stopped_.load(std::memory_order_relaxed)) {
    step();
  }
  LLM_ENGINE_LOG("deepseek") << "run done" << std::endl;
}

void DeepSeekRunner::stop() {
  stopped_.store(true, std::memory_order_relaxed);
}

void DeepSeekRunner::step() {
  int slots = config_.max_num_seqs - in_flight_count_.load(std::memory_order_relaxed);
  if (slots <= 0) return;

  std::vector<Sequence*> batch;
  for (int i = 0; i < slots; ++i) {
    Sequence* seq = task_queue_->try_pop();
    if (!seq) break;
    batch.push_back(seq);
  }
  if (batch.empty()) return;

  {
    std::lock_guard lock(in_flight_mutex_);
    for (Sequence* seq : batch) {
      in_flight_[seq->task_id] = InFlightSeq{
          .task_id_str = seq->task_id.id,
          .max_tokens = static_cast<uint32_t>(seq->sampling_params->max_tokens),
          .tokens_received = 0,
          .ignore_eos = seq->sampling_params->ignore_eos,
      };
    }
  }
  in_flight_count_.fetch_add(static_cast<int>(batch.size()), std::memory_order_relaxed);

  model_runner_->run(batch, true);

  for (Sequence* seq : batch) {
    LLM_ENGINE_LOG("deepseek") << "prefill task_id=" << seq->task_id.id
                                << " prompt_len=" << seq->token_ids_.size()
                                << " max_tokens=" << seq->sampling_params->max_tokens << std::endl;
    delete seq;
  }
}

}  // namespace tt::runners
