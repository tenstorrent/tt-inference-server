// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <deque>
#include <memory>
#include <unordered_set>
#include <vector>
#include <unordered_map>

#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/block_manager.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/llm_runner/sampling_params.hpp"

namespace llm_engine {

/**
 * Schedules prefill and decode batches. Each step returns either a prefill-only
 * or a decode-only batch (no mixed batches).
 * Subclasses control which is attempted first (prefill or decode) via should_prefill_first().
 */
class Scheduler {
 public:
  explicit Scheduler(const Config& config, ITaskQueue* task_queue, size_t batch_size);
  virtual ~Scheduler() = default;

  /** @return true if there are no prefill_queue, decode_queue, or in-flight sequences. */
  bool is_finished() const;

  /** Creates a sequence, takes ownership, and enqueues it for prefill. */
  Sequence& add_request(TaskID task_id, std::vector<int64_t> prompt,
                        const SamplingParams& params = SamplingParams());

  /** Enqueues an externally-owned sequence for prefill (prefill_queue). */
  void add(Sequence& seq);

  /** Looks up a sequence by task_id. Returns nullptr if not found. */
  Sequence* find_sequence(TaskID task_id);

  /**
   * Produces the next batch to run.
   * @return Pair of (scheduled sequences, is_prefill). The batch is either
   *         prefill-only or decode-only; should_prefill_first() determines
   *         which is attempted first.
   */
  std::pair<std::vector<Sequence*>, bool> schedule();

  /**
   * Moves a sequence from decode_queue back to prefill_queue and frees its KV cache blocks.
   */
  void preempt(Sequence& seq);

  /**
   * Appends the generated token to each sequence and marks finished / deallocates
   * as needed. Call after the model runner returns token_ids for the batch.
   * @param seqs  The batch that was just run (same order as token_ids).
   * @param token_ids  One token per sequence from the model.
   */
  void postprocess(std::vector<Sequence*>& seqs,
                   const std::vector<int64_t>& token_ids);
  void removeSequence(TaskID task_id);

  bool is_stop_token(int64_t token_id) const { return stop_token_ids_.count(token_id) > 0; }

 protected:
  /**
   * @param decode_count  number of sequences currently in the decode_queue.
   * @param batch_size    maximum batch / decode_queue capacity.
   * @return true if the scheduler should attempt prefill before decode.
   */
  virtual bool should_prefill_first(int decode_count, int batch_size) const = 0;

  /**
   * Maximum number of sequences to prefill in one step.
   * Default: batch_size (full capacity). Override to limit prefill
   * to available slots when decode sequences should be preserved.
   */
  virtual int max_prefill_seqs(int /*decode_count*/, int batch_size) const {
    return batch_size;
  }

 private:
  int block_size_;
  bool try_schedule_prefill(std::vector<Sequence*>& scheduled_seqs,
                            int& num_seqs, int& num_batched_tokens,
                            int seq_limit);
  void try_schedule_decode(std::vector<Sequence*>& scheduled_seqs,
                           int& num_seqs);

  size_t batch_size_;
  int max_num_batched_tokens_;
  int max_in_flight_count_;
  std::unordered_set<int64_t> stop_token_ids_;
  BlockManager block_manager_;
  ITaskQueue* prefill_queue_;
  std::unordered_map<TaskID, std::unique_ptr<Sequence>> sequences_;
  std::deque<Sequence*> decode_queue_;
  int in_flight_count_ = 0;
};

std::unique_ptr<Scheduler> make_scheduler(const Config& config,
                                          ITaskQueue* task_queue,
                                          size_t batch_size);

}  // namespace llm_engine
