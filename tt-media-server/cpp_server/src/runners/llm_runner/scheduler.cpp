// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "runners/llm_runner/scheduler.hpp"

#include <cassert>

#include "profiling/tracy.hpp"
#include "runners/llm_runner/max_occupancy_scheduler.hpp"
#include "runners/llm_runner/prefill_first_scheduler.hpp"

namespace llm_engine {

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
    : blockSize(config.kvcache_block_size),
      maxInFlightCount(maxInFlightCount),
      maxNumBatchedTokens(config.max_num_batched_tokens),
      stopTokenIds(config.stop_token_ids.begin(), config.stop_token_ids.end()),
      blockManager(config.num_kvcache_blocks, config.kvcache_block_size),
      prefillQueue(taskQueue) {}

bool Scheduler::isFinished() const {
  return prefillQueue->empty() && decodeQueue.empty();
}

Sequence& Scheduler::addRequest(TaskID taskId, std::vector<int64_t> prompt,
                                const SamplingParams& params) {
  auto seq = std::make_unique<Sequence>(std::move(taskId), blockSize,
                                        std::move(prompt), params);
  auto id = seq->task_id;
  add(*seq);
  sequences[id] = std::move(seq);
  return *sequences[id].get();
}

void Scheduler::add(Sequence& seq) { prefillQueue->push(seq); }

Sequence* Scheduler::findSequence(TaskID taskId) {
  auto it = sequences.find(taskId);
  return it != sequences.end() ? it->second.get() : nullptr;
}

bool Scheduler::trySchedulePrefill(std::vector<Sequence*>& scheduledSeqs,
                                   int& numSeqs, int& numBatchedTokens,
                                   int seqLimit) {
  while (numSeqs < seqLimit) {
    auto seq = prefillQueue->try_pop();
    if (!seq) break;

    if (numBatchedTokens + static_cast<int>(seq->size()) >
            maxNumBatchedTokens ||
        !blockManager.can_allocate(*seq)) {
      prefillQueue->push(*seq);
      delete seq;
      break;
    }

    numSeqs += 1;
    blockManager.allocate(*seq);
    numBatchedTokens += static_cast<int>(seq->size() - seq->numCachedTokens_);
    auto id = seq->task_id;
    sequences[id] = std::make_unique<Sequence>(std::move(*seq));
    scheduledSeqs.push_back(sequences[id].get());
    delete seq;
  }
  return !scheduledSeqs.empty();
}

void Scheduler::tryScheduleDecode(std::vector<Sequence*>& scheduledSeqs,
                                  int& numSeqs) {
  while (!decodeQueue.empty() && numSeqs < maxInFlightCount) {
    Sequence* seq = decodeQueue.front();
    decodeQueue.pop_front();
    auto selfPreempt = false;
    while (!blockManager.can_append(*seq)) {
      if (!decodeQueue.empty()) {
        preempt(*decodeQueue.back());
        decodeQueue.pop_back();
      } else {
        preempt(*seq);
        selfPreempt = true;
        break;
      }
    }
    if (!selfPreempt && blockManager.can_append(*seq)) {
      numSeqs += 1;
      blockManager.may_append(*seq);
      scheduledSeqs.push_back(seq);
    }
  }
}

std::pair<std::vector<Sequence*>, bool> Scheduler::schedule() {
  std::vector<Sequence*> scheduledSeqs;
  if (prefillQueue->empty() && decodeQueue.empty()) {
    auto seq = prefillQueue->receive();
    if (seq) {
      prefillQueue->push(*seq);
      delete seq;
    }
  }
  int numSeqs = 0;
  int numBatchedTokens = 0;

  int decodeCount = static_cast<int>(decodeQueue.size());

  bool shouldPrefill = !prefillQueue->empty() &&
                       shouldPrefillFirst(decodeCount, maxInFlightCount);

  if (shouldPrefill) {
    int seqLimit = maxPrefillSeqs(decodeCount, maxInFlightCount);
    if (seqLimit > 0 && trySchedulePrefill(scheduledSeqs, numSeqs,
                                           numBatchedTokens, seqLimit)) {
      return {scheduledSeqs, true};
    }
  }

  tryScheduleDecode(scheduledSeqs, numSeqs);
  return {scheduledSeqs, false};
}

void Scheduler::preempt(Sequence& seq) {
  ZoneScopedN("Scheduler::preempt");
  seq.status_ = SequenceStatus::WAITING;
  blockManager.deallocate(seq);
  prefillQueue->push(seq);
  sequences.erase(seq.task_id);
}

void Scheduler::postprocess(std::vector<Sequence*>& seqs,
                            const std::vector<int64_t>& tokenIds) {
  ZoneScopedN("Scheduler::postprocess");
  for (size_t i = 0; i < seqs.size(); ++i) {
    Sequence* seq = seqs[i];
    int64_t tokenId = tokenIds[i];
    seq->appendToken(tokenId);

    bool isStopToken = stopTokenIds.count(tokenId) > 0;
    bool reachedMaxTokens =
        seq->sampling_params->max_tokens.has_value() &&
        seq->numCompletionTokens() >=
            static_cast<size_t>(seq->sampling_params->max_tokens.value());
    bool finished =
        (!seq->sampling_params->ignore_eos && isStopToken) || reachedMaxTokens;

    if (finished) {
      seq->status_ = SequenceStatus::FINISHED;
      blockManager.deallocate(*seq);
    } else {
      seq->status_ = SequenceStatus::RUNNING;
      decodeQueue.push_back(seq);
    }
  }
}

void Scheduler::removeSequence(TaskID taskId) { sequences.erase(taskId); }
}  // namespace llm_engine
