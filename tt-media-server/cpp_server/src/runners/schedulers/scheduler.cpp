// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/schedulers/scheduler.hpp"

#include <algorithm>

#include "profiling/tracy.hpp"
#include "runners/schedulers/max_occupancy_scheduler.hpp"
#include "runners/schedulers/prefill_first_scheduler.hpp"

namespace tt::runners::schedulers {

using Sequence = tt::domain::Sequence;
using SamplingParams = tt::domain::SamplingParams;
using SequenceStatus = tt::domain::SequenceStatus;

using Config = tt::config::LLMConfig;
using SchedulingPolicy = tt::config::SchedulingPolicy;

std::unique_ptr<Scheduler> makeScheduler(const Config& config,
                                         tt::ipc::ITaskQueue* taskQueue,
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

Scheduler::Scheduler(const Config& config, tt::ipc::ITaskQueue* taskQueue,
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

Sequence& Scheduler::addRequest(uint32_t taskId, std::vector<int64_t> prompt,
                                const SamplingParams& params) {
  auto seq =
      std::make_unique<Sequence>(std::move(taskId), static_cast<int>(blockSize),
                                 std::move(prompt), params);
  auto id = seq->taskId;
  add(*seq);
  sequences[id] = std::move(seq);
  return *sequences[id].get();
}

void Scheduler::add(Sequence& seq) { prefillQueue->push(seq); }

Sequence* Scheduler::findSequence(uint32_t taskId) {
  auto it = sequences.find(taskId);
  return it != sequences.end() ? it->second.get() : nullptr;
}

bool Scheduler::trySchedulePrefill(std::vector<Sequence*>& scheduledSeqs,
                                   size_t& numSeqs, size_t& numBatchedTokens,
                                   size_t seqLimit) {
  while (numSeqs < seqLimit) {
    auto seq = prefillQueue->tryPop();
    if (!seq) break;

    if (pendingAborts.erase(seq->taskId)) {
      sequences.erase(seq->taskId);
      seq.reset();
      continue;
    }

    if (numBatchedTokens + seq->size() > maxNumBatchedTokens ||
        !blockManager.allocate(*seq)) {
      prefillQueue->push(*seq);
      seq.reset();
      break;
    }
    numSeqs += 1;
    numBatchedTokens += seq->size() - seq->getNumCachedTokens();
    auto id = seq->taskId;
    sequences[id] = std::move(seq);
    scheduledSeqs.push_back(sequences[id].get());
  }
  return !scheduledSeqs.empty();
}

void Scheduler::tryScheduleDecode(std::vector<Sequence*>& scheduledSeqs,
                                  size_t& numSeqs) {
  while (!decodeQueue.empty() && numSeqs < maxInFlightCount) {
    Sequence* seq = decodeQueue.front();
    decodeQueue.pop_front();
    auto selfPreempt = false;
    while (!blockManager.canAppend(*seq)) {
      if (!decodeQueue.empty()) {
        preempt(*decodeQueue.back());
        decodeQueue.pop_back();
      } else {
        preempt(*seq);
        selfPreempt = true;
        break;
      }
    }
    if (!selfPreempt && blockManager.canAppend(*seq)) {
      numSeqs += 1;
      blockManager.mayAppend(*seq);
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
      seq.reset();
    }
  }
  size_t numSeqs = 0;
  size_t numBatchedTokens = 0;

  size_t decodeCount = decodeQueue.size();

  bool shouldPrefill = !prefillQueue->empty() &&
                       shouldPrefillFirst(decodeCount, maxInFlightCount);

  if (shouldPrefill) {
    size_t seqLimit = maxPrefillSeqs(decodeCount, maxInFlightCount);
    if (seqLimit > 0 && trySchedulePrefill(scheduledSeqs, numSeqs,
                                           numBatchedTokens, seqLimit)) {
      return {scheduledSeqs, true};
    }
  }

  tryScheduleDecode(scheduledSeqs, numSeqs);

  // Trim stale pending_aborts_ entries.  When the prefill queue is empty,
  // remaining entries can never match a dequeued sequence — they are leftovers
  // from broadcast aborts to workers that don't own those tasks.
  if (prefillQueue->empty() && pendingAborts.size() > maxInFlightCount) {
    pendingAborts.clear();
  }

  return {scheduledSeqs, false};
}

void Scheduler::preempt(Sequence& seq) {
  ZoneScopedN("Scheduler::preempt");
  seq.setStatus(SequenceStatus::WAITING);
  blockManager.deallocate(seq);
  prefillQueue->push(seq);
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
        seq->getSamplingParams().max_tokens.has_value() &&
        seq->numCompletionTokens() >=
            static_cast<size_t>(seq->getSamplingParams().max_tokens.value());
    bool finished = (!seq->getSamplingParams().ignore_eos && isStopToken) ||
                    reachedMaxTokens;

    if (finished) {
      seq->setStatus(SequenceStatus::FINISHED);
      blockManager.deallocate(*seq);
    } else {
      seq->setStatus(SequenceStatus::RUNNING);
      decodeQueue.push_back(seq);
    }
  }
}

void Scheduler::removeSequence(uint32_t taskId) { sequences.erase(taskId); }

void Scheduler::abortRequest(uint32_t taskId) {
  auto* seq = findSequence(taskId);

  // If the task isn't tracked or is still waiting, it might have a stale
  // copy in the IPC queue. Mark it for skipping.
  bool isWaiting = seq && seq->getStatus() == SequenceStatus::WAITING;
  if (!seq || isWaiting) {
    if (pendingAborts.size() < 2 * maxInFlightCount) {
      pendingAborts.insert(taskId);
    }
  }

  if (!seq || seq->isAborted() || seq->isFinished()) {
    return;
  }

  seq->setStatus(SequenceStatus::ABORTED);
  blockManager.deallocate(*seq);

  // Remove from decode_queue (O(n) but abort should be rare)
  std::erase_if(decodeQueue, [&](Sequence* s) { return s->taskId == taskId; });
  sequences.erase(taskId);
}
}  // namespace tt::runners::schedulers
