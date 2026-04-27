// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/sampling_params.hpp"
#include "domain/sequence.hpp"
#include "ipc/task_queue.hpp"
#include "runners/llm_runner/block_manager.hpp"

namespace tt::runners::schedulers {

/**
 * Schedules prefill and decode batches. Each step returns either a prefill-only
 * or a decode-only batch (no mixed batches).
 * Subclasses control which is attempted first (prefill or decode) via
 * should_prefill_first().
 */
class Scheduler {
 public:
  explicit Scheduler(const tt::config::LLMConfig& config,
                     ipc::ITaskQueue* taskQueue, size_t maxInFlightCount);
  virtual ~Scheduler() = default;

  /** @return true if there are no prefill_queue, decode_queue, or in-flight
   * sequences. */
  bool isFinished() const;

  /** Creates a sequence, takes ownership, and enqueues it for prefill. */
  tt::domain::Sequence& addRequest(
      uint32_t taskId, std::vector<int64_t> prompt,
      const tt::domain::SamplingParams& params = tt::domain::SamplingParams());

  /** Enqueues an externally-owned sequence for prefill (prefill_queue). */
  void add(tt::domain::Sequence& seq);

  /** Looks up a sequence by task_id. Returns nullptr if not found. */
  tt::domain::Sequence* findSequence(uint32_t taskId);

  /**
   * Produces the next batch to run.
   * @return Pair of (scheduled sequences, is_prefill). The batch is either
   *         prefill-only or decode-only; should_prefill_first() determines
   *         which is attempted first.
   */
  std::pair<std::vector<tt::domain::Sequence*>, bool> schedule();

  /**
   * Moves a sequence from decode_queue back to prefill_queue and frees its KV
   * cache blocks.
   */
  void preempt(tt::domain::Sequence& seq);

  /**
   * Appends the generated token to each sequence and marks finished /
   * deallocates as needed. Call after the model runner returns token_ids for
   * the batch.
   * @param seqs  The batch that was just run (same order as token_ids).
   * @param token_ids  One token per sequence from the model.
   */
  void postprocess(std::vector<tt::domain::Sequence*>& seqs,
                   const std::vector<int64_t>& tokenIds);
  void removeSequence(uint32_t taskId);

  /**
   * Abort a request by task ID. Frees KV cache blocks, removes the sequence
   * from whichever queue it is currently in, and erases it from the sequences
   * map. Idempotent: a second call for the same ID is a no-op.
   */
  void abortRequest(uint32_t taskId);

  bool isStopToken(int64_t tokenId) const {
    return stopTokenIds.count(tokenId) > 0;
  }

  llm_engine::BlockManager& getBlockManager() { return this->blockManager; }

 protected:
  /**
   * @param decode_count  number of sequences currently in the decode_queue.
   * @param max_in_flight_count    maximum batch / decode_queue capacity.
   * @return true if the scheduler should attempt prefill before decode.
   */
  virtual bool shouldPrefillFirst(size_t decodeCount,
                                  size_t maxInFlightCount) const = 0;

  /**
   * Maximum number of sequences to prefill in one step.
   * Default: max_in_flight_count (full capacity). Override to limit prefill
   * to available slots when decode sequences should be preserved.
   */
  virtual size_t maxPrefillSeqs(size_t /*decode_count*/,
                                size_t maxInFlightCount) const {
    return maxInFlightCount;
  }

 private:
  size_t blockSize;
  bool trySchedulePrefill(std::vector<tt::domain::Sequence*>& scheduledSeqs,
                          size_t& numSeqs, size_t& numBatchedTokens,
                          size_t seqLimit);
  void tryScheduleDecode(std::vector<tt::domain::Sequence*>& scheduledSeqs,
                         size_t& numSeqs);

  size_t maxInFlightCount;
  size_t maxNumBatchedTokens;
  std::unordered_set<int64_t> stopTokenIds;
  llm_engine::BlockManager blockManager;
  ipc::ITaskQueue* prefillQueue;
  std::unordered_map<uint32_t, std::unique_ptr<tt::domain::Sequence>> sequences;
  std::deque<tt::domain::Sequence*> decodeQueue;
  // IDs aborted before their copy was dequeued from the prefill queue.
  // Checked in trySchedulePrefill to skip stale copies.
  std::unordered_set<uint32_t> pendingAborts;
};

std::unique_ptr<Scheduler> makeScheduler(const tt::config::LLMConfig& config,
                                         ipc::ITaskQueue* taskQueue,
                                         size_t maxInFlightCount);

}  // namespace tt::runners::schedulers
