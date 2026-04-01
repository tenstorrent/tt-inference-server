// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "config/runner_config.hpp"
#include "runners/llm_runner/block_manager.hpp"
#include "runners/llm_runner/sampling_params.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/task_queue.hpp"

namespace llm_engine {

/**
 * Schedules prefill and decode batches. Each step returns either a prefill-only
 * or a decode-only batch (no mixed batches).
 * Subclasses control which is attempted first (prefill or decode) via
 * should_prefill_first().
 */
class Scheduler {
 public:
  explicit Scheduler(const tt::config::LLMConfig& config, ITaskQueue* taskQueue,
                     size_t maxInFlightCount);
  virtual ~Scheduler() = default;

  /** @return true if there are no prefill_queue, decode_queue, or in-flight
   * sequences. */
  bool isFinished() const;

  /** Creates a sequence, takes ownership, and enqueues it for prefill. */
  Sequence& addRequest(TaskID taskId, std::vector<int64_t> prompt,
                       const SamplingParams& params = SamplingParams());

  /** Enqueues an externally-owned sequence for prefill (prefill_queue). */
  void add(Sequence& seq);

  /** Looks up a sequence by task_id. Returns nullptr if not found. */
  Sequence* findSequence(TaskID taskId);

  /**
   * Produces the next batch to run.
   * @return Pair of (scheduled sequences, is_prefill). The batch is either
   *         prefill-only or decode-only; should_prefill_first() determines
   *         which is attempted first.
   */
  std::pair<std::vector<Sequence*>, bool> schedule();

  /**
   * Moves a sequence from decode_queue back to prefill_queue and frees its KV
   * cache blocks.
   */
  void preempt(Sequence& seq);

  /**
   * Appends the generated token to each sequence and marks finished /
   * deallocates as needed. Call after the model runner returns token_ids for
   * the batch.
   * @param seqs  The batch that was just run (same order as token_ids).
   * @param token_ids  One token per sequence from the model.
   */
  void postprocess(std::vector<Sequence*>& seqs,
                   const std::vector<int64_t>& tokenIds);
  void removeSequence(TaskID taskId);

  /**
   * Abort a request by task ID. Frees KV cache blocks, removes the sequence
   * from whichever queue it is currently in, and erases it from the sequences
   * map. Idempotent: a second call for the same ID is a no-op.
   */
  void abortRequest(TaskID taskId);

  bool isStopToken(int64_t tokenId) const {
    return stop_token_ids_.count(tokenId) > 0;
  }

  BlockManager& blockManager() { return block_manager_; }

 protected:
  /**
   * @param decode_count  number of sequences currently in the decode_queue.
   * @param max_in_flight_count    maximum batch / decode_queue capacity.
   * @return true if the scheduler should attempt prefill before decode.
   */
  virtual bool shouldPrefillFirst(int decodeCount,
                                  int maxInFlightCount) const = 0;

  /**
   * Maximum number of sequences to prefill in one step.
   * Default: max_in_flight_count (full capacity). Override to limit prefill
   * to available slots when decode sequences should be preserved.
   */
  virtual int maxPrefillSeqs(int /*decode_count*/, int maxInFlightCount) const {
    return maxInFlightCount;
  }

 private:
  int block_size_;
  bool trySchedulePrefill(std::vector<Sequence*>& scheduledSeqs, int& numSeqs,
                          int& numBatchedTokens, int seqLimit);
  void tryScheduleDecode(std::vector<Sequence*>& scheduledSeqs, int& numSeqs);

  size_t max_in_flight_count_;
  int max_num_batched_tokens_;
  std::unordered_set<int64_t> stop_token_ids_;
  BlockManager block_manager_;
  ITaskQueue* prefill_queue_;
  std::unordered_map<TaskID, std::unique_ptr<Sequence>> sequences_;
  std::deque<Sequence*> decode_queue_;
  // IDs aborted before their copy was dequeued from the prefill queue.
  // Checked in trySchedulePrefill to skip stale copies.
  std::unordered_set<TaskID> pending_aborts_;
};

std::unique_ptr<Scheduler> makeScheduler(const tt::config::LLMConfig& config,
                                         ITaskQueue* taskQueue,
                                         size_t maxInFlightCount);

}  // namespace llm_engine
