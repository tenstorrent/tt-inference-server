#include "runners/llm_runner/scheduler.hpp"
#include "runners/llm_runner/debug.hpp"

#include <cassert>


namespace llm_engine {

Scheduler::Scheduler(const Config& config, ITaskQueue* task_queue)
    : max_num_seqs_(config.max_num_seqs),
      max_num_batched_tokens_(config.max_num_batched_tokens),
      stop_token_ids_(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      block_manager_(config.num_kvcache_blocks, config.kvcache_block_size),
      waiting_(task_queue) {}

bool Scheduler::is_finished() const {
  return waiting_->empty() && running_.empty() && in_flight_count_ == 0;
}

Sequence& Scheduler::add_request(std::vector<int64_t> prompt,
                                  const SamplingParams& params) {
  auto seq = std::make_unique<Sequence>(std::move(prompt), params);
  Sequence& ref = *seq;
  auto id = seq->task_id;
  add(ref);
  sequences_[id] = std::move(seq);
  return *sequences_[id].get();
}

void Scheduler::add(Sequence& seq) {
  LLM_ENGINE_LOG("scheduler") << "add task_id=" << seq.task_id << " len=" << seq.size() << std::endl;
  waiting_->push(seq);
}

Sequence* Scheduler::find_sequence(TaskID task_id) {
  auto it = sequences_.find(task_id);
  return it != sequences_.end() ? it->second.get() : nullptr;
}

std::pair<std::vector<Sequence*>, bool> Scheduler::schedule() {
  std::vector<Sequence*> scheduled_seqs;
  int num_seqs = 0;
  int num_batched_tokens = 0;

  // --- Prefill: pop from task queue ---
  while (num_seqs < max_num_seqs_) {
    auto seq = waiting_->try_pop();
    if (!seq) {
      break;  // Queue empty
    }

    if (num_batched_tokens + static_cast<int>(seq->size()) >
            max_num_batched_tokens_ ||
        !block_manager_.can_allocate(*seq)) {
      // Can't handle this sequence -- push it back and stop prefilling.
      waiting_->push(*seq);
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

  if (!scheduled_seqs.empty()) {
    LLM_ENGINE_LOG("scheduler") << "schedule prefill n=" << scheduled_seqs.size()
                             << " batched_tokens=" << num_batched_tokens << std::endl;
    return {scheduled_seqs, true};
  }

  // --- Decode: process running sequences ---
  while (!running_.empty() && num_seqs < max_num_seqs_) {
    Sequence* seq = running_.front();
    running_.pop_front();
    auto self_preempt = false;
    while (!block_manager_.can_append(*seq)) {
      if (!running_.empty()) {
        preempt(*running_.back());
        running_.pop_back();
      } else {
        preempt(*seq);
        self_preempt = true;
        break;
      }
    }
    if (block_manager_.can_append(*seq) && !self_preempt) {
      num_seqs += 1;
      block_manager_.may_append(*seq);
      scheduled_seqs.push_back(seq);
    }
  }
  for (Sequence* seq : scheduled_seqs) {
    seq->status_ = SequenceStatus::IN_FLIGHT;
    ++in_flight_count_;
  }
  LLM_ENGINE_LOG("scheduler") << "schedule decode n=" << scheduled_seqs.size()
                             << " scheduled_seqs=" << (scheduled_seqs.empty() ? "none" : scheduled_seqs[0]->task_id.id)
                             << " in_flight=" << in_flight_count_ << std::endl;

  return {scheduled_seqs, false};
}

void Scheduler::preempt(Sequence& seq) {
  LLM_ENGINE_LOG("scheduler") << "preempt task_id=" << seq.task_id << std::endl;
  seq.status_ = SequenceStatus::WAITING;
  block_manager_.deallocate(seq);
  waiting_->push(seq);
}

void Scheduler::postprocess(std::vector<Sequence*>& seqs,
                            const std::vector<int64_t>& token_ids) {
  for (size_t i = 0; i < seqs.size(); ++i) {
    Sequence* seq = seqs[i];
    int64_t token_id = token_ids[i];
    assert(seq->status_ == SequenceStatus::IN_FLIGHT);

    seq->append_token(token_id);

    bool is_stop_token = stop_token_ids_.count(token_id) > 0;
    bool finished =
        (!seq->ignore_eos && is_stop_token) ||
        seq->num_completion_tokens() >= static_cast<size_t>(seq->max_tokens);

    if (finished) {
      LLM_ENGINE_LOG("scheduler") << "postprocess task_id=" << seq->task_id << " finished"
          << " (stop_token=" << is_stop_token << " max_tokens="
          << (seq->num_completion_tokens() >= static_cast<size_t>(seq->max_tokens)) << ")" << std::endl;
      seq->status_ = SequenceStatus::FINISHED;
      block_manager_.deallocate(*seq);
      --in_flight_count_;
    } else {
      seq->status_ = SequenceStatus::RUNNING;
      running_.push_back(seq);
      --in_flight_count_;
    }
  }

}

void Scheduler::removeSequence(TaskID task_id) {
  sequences_.erase(task_id);
}
}  // namespace llm_engine
