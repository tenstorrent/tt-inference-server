#include "llm_engine/engine/scheduler.hpp"
#include "llm_engine/engine/debug.hpp"
#include <algorithm>
#include <cassert>

namespace llm_engine {

Scheduler::Scheduler(const Config& config)
    : max_num_seqs_(config.max_num_seqs),
      max_num_batched_tokens_(config.max_num_batched_tokens),
      eos_(config.eos),
      block_manager_(config.num_kvcache_blocks, config.kvcache_block_size) {}

bool Scheduler::is_finished() const {
  return waiting_.empty() && running_.empty();
}

void Scheduler::add(Sequence& seq) {
  LLM_ENGINE_LOG("scheduler") << "add seq_id=" << seq.seq_id << " len=" << seq.size() << std::endl;
  waiting_.push_back(&seq);
}

std::pair<std::vector<Sequence*>, bool> Scheduler::schedule() {
  std::vector<Sequence*> scheduled_seqs;
  int num_seqs = 0;
  int num_batched_tokens = 0;

  while (!waiting_.empty() && num_seqs < max_num_seqs_) {
    Sequence* seq = waiting_.front();
    if (num_batched_tokens + static_cast<int>(seq->size()) >
            max_num_batched_tokens_ ||
        !block_manager_.can_allocate(*seq)) {
      break;
    }
    num_seqs += 1;
    block_manager_.allocate(*seq);
    num_batched_tokens +=
        static_cast<int>(seq->size() - seq->num_cached_tokens_);
    seq->status_ = SequenceStatus::RUNNING;
    waiting_.pop_front();
    running_.push_back(seq);
    scheduled_seqs.push_back(seq);
  }
  if (!scheduled_seqs.empty()) {
    LLM_ENGINE_LOG("scheduler") << "schedule prefill n=" << scheduled_seqs.size()
                              << " batched_tokens=" << num_batched_tokens << std::endl;
    return {scheduled_seqs, true};
  }

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
  for (auto it = scheduled_seqs.rbegin(); it != scheduled_seqs.rend(); ++it) {
    running_.push_front(*it);
  }
  LLM_ENGINE_LOG("scheduler") << "schedule decode n=" << scheduled_seqs.size() << std::endl;
  return {scheduled_seqs, false};
}

void Scheduler::preempt(Sequence& seq) {
  LLM_ENGINE_LOG("scheduler") << "preempt seq_id=" << seq.seq_id << std::endl;
  seq.status_ = SequenceStatus::WAITING;
  block_manager_.deallocate(seq);
  waiting_.push_front(&seq);
}

void Scheduler::postprocess(std::vector<Sequence*>& seqs,
                            const std::vector<int64_t>& token_ids) {
  for (size_t i = 0; i < seqs.size(); ++i) {
    Sequence* seq = seqs[i];
    int64_t token_id = token_ids[i];
    seq->append_token(token_id);
    if ((!seq->ignore_eos && token_id == eos_) ||
        seq->num_completion_tokens() == static_cast<size_t>(seq->max_tokens)) {
      LLM_ENGINE_LOG("scheduler") << "postprocess seq_id=" << seq->seq_id << " finished"
                               << " (eos=" << (token_id == eos_) << " max_tokens="
                               << (seq->num_completion_tokens() == static_cast<size_t>(seq->max_tokens)) << ")" << std::endl;
      seq->status_ = SequenceStatus::FINISHED;
      block_manager_.deallocate(*seq);
      running_.erase(
          std::find(running_.begin(), running_.end(), seq));
    }
  }
}

}  // namespace llm_engine
