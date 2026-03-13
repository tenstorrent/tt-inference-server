// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/llm_runner/scheduler.hpp"
#include "runners/llm_runner/prefill_first_scheduler.hpp"
#include "runners/llm_runner/max_occupancy_scheduler.hpp"
#include "runners/llm_runner/debug.hpp"
#include "profiling/tracy.hpp"

#include <cassert>


namespace llm_engine {

std::unique_ptr<Scheduler> make_scheduler(const Config& config,
                                          ITaskQueue* task_queue) {
  switch (config.scheduling_policy) {
    case SchedulingPolicy::MAX_OCCUPANCY:
      return std::make_unique<MaxOccupancyScheduler>(config, task_queue);
    case SchedulingPolicy::PREFILL_FIRST:
    default:
      return std::make_unique<PrefillFirstScheduler>(config, task_queue);
  }
}

Scheduler::Scheduler(const Config& config, ITaskQueue* task_queue)
    : block_size_(config.kvcache_block_size),
      max_num_seqs_(config.max_num_seqs),
      max_num_batched_tokens_(config.max_num_batched_tokens),
      max_in_flight_count_(config.max_in_flight_count),
      stop_token_ids_(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      block_manager_(config.num_kvcache_blocks, config.kvcache_block_size),
      prefill_queue_(task_queue) {}

bool Scheduler::is_finished() const {
  return prefill_queue_->empty() && decode_queue_.empty() && in_flight_count_ == 0;
}

Sequence& Scheduler::add_request(TaskID task_id, std::vector<int64_t> prompt,
                                  const SamplingParams& params) {
  auto seq = std::make_unique<Sequence>(std::move(task_id), block_size_, std::move(prompt), params);
  auto id = seq->task_id;
  add(*seq);
  sequences_[id] = std::move(seq);
  return *sequences_[id].get();
}

void Scheduler::add(Sequence& seq) {
  LLM_ENGINE_LOG("scheduler") << "add task_id=" << seq.task_id << " len=" << seq.size() << std::endl;
  prefill_queue_->push(seq);
}

Sequence* Scheduler::find_sequence(TaskID task_id) {
  auto it = sequences_.find(task_id);
  return it != sequences_.end() ? it->second.get() : nullptr;
}


bool Scheduler::try_schedule_prefill(std::vector<Sequence*>& scheduled_seqs,
                                     int& num_seqs, int& num_batched_tokens,
                                     int seq_limit) {
  while (num_seqs < seq_limit && in_flight_count_ < max_in_flight_count_) {
    auto seq = prefill_queue_->try_pop();
    if (!seq) break;

    if (num_batched_tokens + static_cast<int>(seq->size()) >
            max_num_batched_tokens_ ||
        !block_manager_.can_allocate(*seq)) {
      prefill_queue_->push(*seq);
      delete seq;
      break;
    }

    num_seqs += 1;
    block_manager_.allocate(*seq);
    num_batched_tokens +=
        static_cast<int>(seq->size() - seq->num_cached_tokens_);
    seq->status_ = SequenceStatus::IN_FLIGHT;
    ++in_flight_count_;
    auto id = seq->task_id;
    sequences_[id] = std::make_unique<Sequence>(std::move(*seq));
    scheduled_seqs.push_back(sequences_[id].get());
    delete seq;
  }
  return !scheduled_seqs.empty();
}

void Scheduler::try_schedule_decode(std::vector<Sequence*>& scheduled_seqs,
                                    int& num_seqs) {
  while (!decode_queue_.empty() && num_seqs < max_num_seqs_ && in_flight_count_ < max_in_flight_count_) {
    Sequence* seq = decode_queue_.front();
    decode_queue_.pop_front();
    auto self_preempt = false;
    while (!block_manager_.can_append(*seq)) {
      if (!decode_queue_.empty()) {
        preempt(*decode_queue_.back());
        decode_queue_.pop_back();
      } else {
        preempt(*seq);
        self_preempt = true;
        break;
      }
    }
    if (block_manager_.can_append(*seq) && !self_preempt) {
      num_seqs += 1;
      block_manager_.may_append(*seq);
      seq->status_ = SequenceStatus::IN_FLIGHT;
      ++in_flight_count_;
      scheduled_seqs.push_back(seq);
    }
  }
}

std::pair<std::vector<Sequence*>, bool> Scheduler::schedule() { 
  std::vector<Sequence*> scheduled_seqs;
  if (prefill_queue_->empty() && decode_queue_.empty() && in_flight_count_ == 0) {
    auto seq =prefill_queue_->receive();
    if (seq) {
      prefill_queue_->push(*seq);
      delete seq;
    }
  }
  int num_seqs = 0;
  int num_batched_tokens = 0;

  int decode_count = static_cast<int>(decode_queue_.size());
  bool should_prefill = !prefill_queue_->empty() && should_prefill_first(decode_count, max_num_seqs_);

  if (should_prefill) {
    int seq_limit = max_prefill_seqs(decode_count, max_num_seqs_);
    if (seq_limit > 0 &&
        try_schedule_prefill(scheduled_seqs, num_seqs, num_batched_tokens, seq_limit)) {
      LLM_ENGINE_LOG("scheduler")
          << "schedule prefill n=" << scheduled_seqs.size()
          << " batched_tokens=" << num_batched_tokens << std::endl;
      return {scheduled_seqs, true};
    }
  }

  try_schedule_decode(scheduled_seqs, num_seqs);
  LLM_ENGINE_LOG("scheduler")
      << "schedule decode n=" << scheduled_seqs.size()
      << " in_flight=" << in_flight_count_ << std::endl;

  return {scheduled_seqs, false};
}

void Scheduler::preempt(Sequence& seq) {
  ZoneScopedN("Scheduler::preempt");
  LLM_ENGINE_LOG("scheduler") << "preempt task_id=" << seq.task_id << std::endl;
  seq.status_ = SequenceStatus::WAITING;
  block_manager_.deallocate(seq);
  prefill_queue_->push(seq);
}

void Scheduler::postprocess(std::vector<Sequence*>& seqs,
                            const std::vector<int64_t>& token_ids) {
  ZoneScopedN("Scheduler::postprocess");
  for (size_t i = 0; i < seqs.size(); ++i) {
    Sequence* seq = seqs[i];
    int64_t token_id = token_ids[i];
    assert(seq->status_ == SequenceStatus::IN_FLIGHT);

    seq->append_token(token_id);

    bool is_stop_token = stop_token_ids_.count(token_id) > 0;
    bool reached_max_tokens = false;
    if (seq->sampling_params->max_tokens.has_value()) {
      reached_max_tokens = seq->num_completion_tokens() >=
          static_cast<size_t>(seq->sampling_params->max_tokens.value());
    }
    bool finished =
        (!seq->sampling_params->ignore_eos && is_stop_token) ||
        reached_max_tokens;

    if (finished) {
      LLM_ENGINE_LOG("scheduler") << "postprocess task_id=" << seq->task_id << " finished"
          << " (stop_token=" << is_stop_token << " max_tokens="
          << reached_max_tokens << ")" << std::endl;
      seq->status_ = SequenceStatus::FINISHED;
      block_manager_.deallocate(*seq);
      --in_flight_count_;
    } else {
      seq->status_ = SequenceStatus::RUNNING;
      decode_queue_.push_back(seq);
      --in_flight_count_;
    }
  }

}

void Scheduler::removeSequence(TaskID task_id) {
  sequences_.erase(task_id);
}
}  // namespace llm_engine
