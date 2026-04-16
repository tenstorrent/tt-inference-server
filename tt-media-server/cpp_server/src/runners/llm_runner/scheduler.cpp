// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/llm_runner/scheduler.hpp"

#include <algorithm>

#include "profiling/tracy.hpp"
#include "runners/llm_runner/max_occupancy_scheduler.hpp"
#include "runners/llm_runner/prefill_first_scheduler.hpp"

namespace tt::runners::llm_engine {

using Config = tt::config::LLMConfig;
using SchedulingPolicy = tt::config::SchedulingPolicy;

std::unique_ptr<Scheduler> makeScheduler(const Config& config,
                                         ITaskQueue* taskQueue,
                                         size_t maxInFlightCount) {
  switch (config.scheduling_policy) {
    case SchedulingPolicy::MAX_OCCUPANCY:
      return std::make_unique<MaxOccupancyScheduler>(config, taskQueue,
                                                     maxInFlightCount);
    case SchedulingPolicy::PREFILL_FIRST:
    default:
      return std::make_unique<PrefillFirstScheduler>(config, taskQueue,
                                                     maxInFlightCount);
  }
}

Scheduler::Scheduler(const Config& config, ITaskQueue* taskQueue,
                     size_t maxInFlightCount)
    : block_size_(config.kvcache_block_size),
      max_in_flight_count_(maxInFlightCount),
      max_num_batched_tokens_(config.max_num_batched_tokens),
      stop_token_ids_(config.stop_token_ids.begin(),
                      config.stop_token_ids.end()),
      block_manager_(config.num_kvcache_blocks, config.kvcache_block_size),
      prefill_queue_(taskQueue) {}

bool Scheduler::isFinished() const {
  return prefill_queue_->empty() && decode_queue_.empty();
}

Sequence& Scheduler::addRequest(uint32_t taskId, std::vector<int64_t> prompt,
                                const SamplingParams& params) {
  auto seq = std::make_unique<Sequence>(std::move(taskId),
                                        static_cast<int>(block_size_),
                                        std::move(prompt), params);
  auto id = seq->taskId;
  add(*seq);
  sequences_[id] = std::move(seq);
  return *sequences_[id].get();
}

void Scheduler::add(Sequence& seq) { prefill_queue_->push(seq); }

Sequence* Scheduler::findSequence(uint32_t taskId) {
  auto it = sequences_.find(taskId);
  return it != sequences_.end() ? it->second.get() : nullptr;
}

bool Scheduler::trySchedulePrefill(std::vector<Sequence*>& scheduledSeqs,
                                   size_t& numSeqs, size_t& numBatchedTokens,
                                   size_t seqLimit) {
  while (numSeqs < seqLimit) {
    auto seq = prefill_queue_->tryPop();
    if (!seq) break;

    if (pending_aborts_.erase(seq->taskId)) {
      sequences_.erase(seq->taskId);
      seq.reset();
      continue;
    }

    if (numBatchedTokens + seq->size() > max_num_batched_tokens_ ||
        !block_manager_.allocate(*seq)) {
      prefill_queue_->push(*seq);
      seq.reset();
      break;
    }
    numSeqs += 1;
    numBatchedTokens += seq->size() - seq->getNumCachedTokens();
    auto id = seq->taskId;
    sequences_[id] = std::move(seq);
    scheduledSeqs.push_back(sequences_[id].get());
  }
  return !scheduledSeqs.empty();
}

void Scheduler::tryScheduleDecode(std::vector<Sequence*>& scheduledSeqs,
                                  size_t& numSeqs) {
  while (!decode_queue_.empty() && numSeqs < max_in_flight_count_) {
    Sequence* seq = decode_queue_.front();
    decode_queue_.pop_front();
    auto selfPreempt = false;
    while (!block_manager_.canAppend(*seq)) {
      if (!decode_queue_.empty()) {
        preempt(*decode_queue_.back());
        decode_queue_.pop_back();
      } else {
        preempt(*seq);
        selfPreempt = true;
        break;
      }
    }
    if (!selfPreempt && block_manager_.canAppend(*seq)) {
      numSeqs += 1;
      block_manager_.mayAppend(*seq);
      scheduledSeqs.push_back(seq);
    }
  }
}

std::pair<std::vector<Sequence*>, bool> Scheduler::schedule() {
  std::vector<Sequence*> scheduledSeqs;
  if (prefill_queue_->empty() && decode_queue_.empty()) {
    auto seq = prefill_queue_->receive();
    if (seq) {
      prefill_queue_->push(*seq);
      seq.reset();
    }
  }
  size_t numSeqs = 0;
  size_t numBatchedTokens = 0;

  size_t decodeCount = decode_queue_.size();

  bool shouldPrefill = !prefill_queue_->empty() &&
                       shouldPrefillFirst(decodeCount, max_in_flight_count_);

  if (shouldPrefill) {
    size_t seqLimit = maxPrefillSeqs(decodeCount, max_in_flight_count_);
    if (seqLimit > 0 && trySchedulePrefill(scheduledSeqs, numSeqs,
                                           numBatchedTokens, seqLimit)) {
      return {scheduledSeqs, true};
    }
  }

  tryScheduleDecode(scheduledSeqs, numSeqs);

  // Trim stale pending_aborts_ entries.  When the prefill queue is empty,
  // remaining entries can never match a dequeued sequence — they are leftovers
  // from broadcast aborts to workers that don't own those tasks.
  if (prefill_queue_->empty() &&
      pending_aborts_.size() > max_in_flight_count_) {
    pending_aborts_.clear();
  }

  return {scheduledSeqs, false};
}

void Scheduler::preempt(Sequence& seq) {
  ZoneScopedN("Scheduler::preempt");
  seq.setStatus(SequenceStatus::WAITING);
  block_manager_.deallocate(seq);
  prefill_queue_->push(seq);
}

void Scheduler::postprocess(std::vector<Sequence*>& seqs,
                            const std::vector<int64_t>& tokenIds) {
  ZoneScopedN("Scheduler::postprocess");
  for (size_t i = 0; i < seqs.size(); ++i) {
    Sequence* seq = seqs[i];
    int64_t tokenId = tokenIds[i];
    seq->appendToken(tokenId);

    bool isStopToken = stop_token_ids_.count(tokenId) > 0;
    bool reachedMaxTokens =
        seq->getSamplingParams().max_tokens.has_value() &&
        seq->numCompletionTokens() >=
            static_cast<size_t>(seq->getSamplingParams().max_tokens.value());
    bool finished = (!seq->getSamplingParams().ignore_eos && isStopToken) ||
                    reachedMaxTokens;

    if (finished) {
      seq->setStatus(SequenceStatus::FINISHED);
      block_manager_.deallocate(*seq);
    } else {
      seq->setStatus(SequenceStatus::RUNNING);
      decode_queue_.push_back(seq);
    }
  }
}

void Scheduler::removeSequence(uint32_t taskId) { sequences_.erase(taskId); }

void Scheduler::abortRequest(uint32_t taskId) {
  auto* seq = findSequence(taskId);

  // If the task isn't tracked or is still waiting, it might have a stale
  // copy in the IPC queue. Mark it for skipping.
  bool isWaiting = seq && seq->getStatus() == SequenceStatus::WAITING;
  if (!seq || isWaiting) {
    if (pending_aborts_.size() < 2 * max_in_flight_count_) {
      pending_aborts_.insert(taskId);
    }
  }

  // If the sequence doesn't exist or is already terminal, we're done.
  if (!seq || seq->isAborted() || seq->isFinished()) {
    return;
  }

  // Clean up resources for active sequences
  seq->setStatus(SequenceStatus::ABORTED);
  block_manager_.deallocate(*seq);

  // Remove from decode_queue (O(n) but abort should be rare)
  std::erase_if(decode_queue_,
                [&](Sequence* s) { return s->taskId == taskId; });
  sequences_.erase(taskId);
}
}  // namespace tt::runners::llm_engine
