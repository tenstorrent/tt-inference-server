#include "runners/llm_runner/scheduler.hpp"
#include "runners/llm_runner/debug.hpp"

#include <cassert>


namespace llm_engine {

std::unique_ptr<ISchedulingStrategy> make_scheduling_strategy(
    SchedulingPolicy policy) {
  switch (policy) {
    case SchedulingPolicy::INTERLEAVED:
      return std::make_unique<InterleavedStrategy>();
    case SchedulingPolicy::PREFILL_FIRST:
    default:
      return std::make_unique<PrefillFirstStrategy>();
  }
}

Scheduler::Scheduler(const Config& config, ITaskQueue* task_queue)
    : block_size_(config.kvcache_block_size),
      max_num_seqs_(config.max_num_seqs),
      max_num_batched_tokens_(config.max_num_batched_tokens),
      stop_token_ids_(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      block_manager_(config.num_kvcache_blocks, config.kvcache_block_size),
      waiting_(task_queue),
      strategy_(make_scheduling_strategy(config.scheduling_policy)) {}

bool Scheduler::is_finished() const {
  return waiting_->empty() && running_.empty() && in_flight_count_ == 0;
}

Sequence& Scheduler::add_request(std::vector<int64_t> prompt,
                                  const SamplingParams& params) {
  auto seq = std::make_unique<Sequence>(block_size_, std::move(prompt), params);
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


bool Scheduler::try_schedule_prefill(std::vector<Sequence*>& scheduled_seqs,
                                     int& num_seqs, int& num_batched_tokens) {
  while (num_seqs < max_num_seqs_) {
    auto seq = waiting_->try_pop();
    if (!seq) break;

    if (num_batched_tokens + static_cast<int>(seq->size()) >
            max_num_batched_tokens_ ||
        !block_manager_.can_allocate(*seq)) {
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
  return !scheduled_seqs.empty();
}

<<<<<<< HEAD
  if (!scheduled_seqs.empty()) {
    LLM_ENGINE_LOG("scheduler") << "schedule prefill n=" << scheduled_seqs.size()
                             << " batched_tokens=" << num_batched_tokens << std::endl;
    return {scheduled_seqs, true};
  }

  // --- Decode: process running sequences ---
=======
void Scheduler::try_schedule_decode(std::vector<Sequence*>& scheduled_seqs,
                                    int& num_seqs) {
>>>>>>> d114fefb (Enable Interleaved Batching)
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
}

std::pair<std::vector<Sequence*>, bool> Scheduler::schedule() {
  std::vector<Sequence*> scheduled_seqs;
  int num_seqs = 0;
  int num_batched_tokens = 0;

  bool prefill_first = strategy_->should_prefill_first(
      !waiting_->empty(), static_cast<int>(running_.size()), max_num_seqs_);

  if (prefill_first) {
    if (try_schedule_prefill(scheduled_seqs, num_seqs, num_batched_tokens)) {
      strategy_->on_step_complete(true);
      LLM_ENGINE_LOG("scheduler")
          << "schedule prefill n=" << scheduled_seqs.size()
          << " batched_tokens=" << num_batched_tokens << std::endl;
      return {scheduled_seqs, true};
    }
  }

  try_schedule_decode(scheduled_seqs, num_seqs);
  if (!scheduled_seqs.empty()) {
    strategy_->on_step_complete(false);
    LLM_ENGINE_LOG("scheduler")
        << "schedule decode n=" << scheduled_seqs.size()
        << " scheduled_seqs=" << scheduled_seqs[0]->task_id.id
        << " in_flight=" << in_flight_count_ << std::endl;
    return {scheduled_seqs, false};
  }

  LLM_ENGINE_LOG("scheduler") << "schedule decode n=0 scheduled_seqs=none"
                               << " in_flight=" << in_flight_count_
                               << std::endl;
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
        (!seq->sampling_params->ignore_eos && is_stop_token) ||
        seq->num_completion_tokens() >= static_cast<size_t>(seq->sampling_params->max_tokens);

    if (finished) {
      LLM_ENGINE_LOG("scheduler") << "postprocess task_id=" << seq->task_id << " finished"
          << " (stop_token=" << is_stop_token << " max_tokens="
          << (seq->num_completion_tokens() >= static_cast<size_t>(seq->sampling_params->max_tokens)) << ")" << std::endl;
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
